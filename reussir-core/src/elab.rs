use rustc_hash::{FxHashMapRand, FxHashSetRand};
use thiserror::Error;
use tracing::{error, trace};
use ustr::Ustr;

use crate::{
    ctx::Context,
    eval::{quote, Environment},
    meta::{CheckEntry, CheckVar, MetaContext, MetaEntry, MetaVar},
    term::{Term, TermPtr},
    utils::{with_span, DBLvl, Icit, Name, Pruning, Span, Spine, WithSpan},
    value::{self, Value, ValuePtr},
    Result,
};

use std::{collections::hash_map::Entry as HMapEntry, rc::Rc};
use std::{collections::hash_set::Entry as HSetEntry, convert::identity};

thread_local! {
    static TERM_PLACEHOLDER: TermPtr = with_span(Term::Hole, Span::default());
}

#[derive(Debug, Clone, Copy, Error)]
pub enum Error {
    #[error("expected type and inferred type mismatch")]
    ExpectedInferredMismatch,
    #[error("expected type and soecified binder type mismatch")]
    LambdaBinderType,
    #[error("failed to elaborate terms into same types")]
    Placeholder,
    #[error("meta variable is already solved")]
    SolvedMeta,
    #[error("meta variable occurs in its own solution")]
    OccursCheck,
    #[error("unsupported unification pattern")]
    Unsupported,
    #[error("escaping rigid variable")]
    EscapingRigid,
}

pub struct Elaborator {
    ctx: Context,
    meta: MetaContext,
}

impl Default for Elaborator {
    fn default() -> Self {
        Self::new()
    }
}

impl Elaborator {
    pub fn new() -> Self {
        Self {
            ctx: Context::new(),
            meta: MetaContext::new(),
        }
    }
    pub fn with_context<F, R>(&mut self, new_ctx: Context, f: F) -> R
    where
        F: FnOnce(&mut Self) -> R,
    {
        let old_ctx = std::mem::replace(&mut self.ctx, new_ctx);
        let res = f(self);
        self.ctx = old_ctx;
        res
    }

    fn unify_placeholder(&mut self, expected: TermPtr, mvar: MetaVar) -> Result<()> {
        match self.meta.get_meta(mvar) {
            MetaEntry::Unsolved { ty, blocking } => {
                trace!(
                    "unifying unconstrained meta {:?} with placeholder {:?}",
                    mvar,
                    expected
                );
                let span = expected.span;
                let solution = self.ctx.close_term(expected, span);
                let solution = Environment::new().evaluate(solution, &self.meta)?;
                let ty = ty.clone();
                let blocking = blocking.clone();
                self.meta
                    .set_meta(mvar, MetaEntry::Solved { val: solution, ty });
                for i in blocking.iter().cloned() {
                    self.retry_check(i)?;
                }
            }
            MetaEntry::Solved { val, .. } => {
                trace!(
                    "unifying solved meta {:?} with placeholder {:?}",
                    mvar,
                    expected
                );
                let lhs = self.ctx.env_mut().evaluate(expected, &self.meta)?;
                let (env, pruning) = self.ctx.env_mut_pruning();
                let meta = &self.meta;
                let rhs = env.app_pruning(val.clone(), pruning, meta, val.span)?;
                self.unify(lhs, rhs, Error::Placeholder)?;
            }
        }
        Ok(())
    }
    fn retry_check(&mut self, var: CheckVar) -> Result<()> {
        trace!("retrying check {var:?}");
        if let CheckEntry::Unchecked {
            ctx,
            term,
            ty,
            meta,
        } = self.meta.get_check(var).clone()
        {
            let ty = self.meta.force(ty.clone())?;
            if let Value::Flex(m, _) = ty.data() {
                trace!("delayed check {var:?} is still blocked by meta {m:?}");
                self.meta.add_blocker(var, *m);
            } else {
                trace!("delayed check {var:?} is no longer blocked");
                self.with_context(ctx, |this| {
                    let term = this.check(term, ty)?;
                    this.unify_placeholder(term.clone(), meta)?;
                    this.meta.set_check(var, CheckEntry::Checked(term));
                    Ok(())
                })?;
            }
        }
        Ok(())
    }
    pub fn check_all(&mut self) -> Result<()> {
        for var in self.meta.check_vars() {
            trace!(
                "checking all delay checkes (crreunt {var:?}, {} in total)",
                self.meta.num_checks()
            );
            let CheckEntry::Unchecked {
                ctx,
                term,
                ty: expected,
                meta,
            } = self.meta.get_check(var).clone()
            else {
                continue;
            };
            self.with_context(ctx.clone(), |this| {
                let inferred = this.infer(term)?;
                let (term, ty) = this.apply_implicit_if_neutral(inferred)?;
                this.meta.set_check(var, CheckEntry::Checked(term.clone()));
                this.unify(expected, ty, Error::ExpectedInferredMismatch)?;
                this.unify_placeholder(term, meta)
            })?;
        }
        Ok(())
    }
    fn evaluated_close_type(&self, value: ValuePtr, span: Span) -> Result<ValuePtr> {
        let term = self
            .ctx
            .close_type(quote(self.ctx.level, value, &self.meta)?, span);
        Environment::new().evaluate(term, &self.meta)
    }
    fn fresh_meta(&mut self, ty: ValuePtr) -> Result<TermPtr> {
        let span = ty.span;
        let closed_ty = self.evaluated_close_type(ty, span)?;
        let mvar = self.meta.new_meta(closed_ty, Default::default());
        trace!("introduced new meta variable {mvar:?}");
        let mvar = with_span(Term::Meta(mvar), span);
        let term = Term::AppPruning(mvar, self.ctx.pruning.clone());
        Ok(with_span(term, span))
    }
    fn apply_all_implicit_args(
        &mut self,
        (mut term, mut ty): (TermPtr, ValuePtr),
    ) -> Result<(TermPtr, ValuePtr)> {
        loop {
            ty = self.meta.force(ty)?;
            match ty.data() {
                Value::Pi(_, Icit::Impl, arg_ty, body) => {
                    let meta = self.fresh_meta(arg_ty.clone())?;
                    let value = self.ctx.env_mut().evaluate(meta.clone(), &self.meta)?;
                    let span = term.span;
                    term = with_span(Term::App(term, meta, Icit::Impl), span);
                    ty = body.apply(value, &self.meta)?;
                    continue;
                }
                _ => return Ok((term, ty)),
            }
        }
    }
    fn apply_implicit_if_neutral(
        &mut self,
        (term, ty): (TermPtr, ValuePtr),
    ) -> Result<(TermPtr, ValuePtr)> {
        if let Term::Lambda(_, Icit::Impl, _) = term.data() {
            Ok((term, ty))
        } else {
            self.apply_all_implicit_args((term, ty))
        }
    }
    pub fn unify_impl(&mut self, lhs: ValuePtr, rhs: ValuePtr) -> Result<()> {
        todo!("unify")
    }
    pub fn unify(&mut self, lhs: ValuePtr, rhs: ValuePtr, err: Error) -> Result<()> {
        self.unify_impl(lhs.clone(), rhs.clone())
            .inspect_err(|e| {
                trace!("failed to unify {lhs:?} with {rhs:?} ({e})");
            })
            .map_err(|_| {
                let lhs = quote(self.ctx.level, lhs, &self.meta)?;
                let rhs = quote(self.ctx.level, rhs, &self.meta)?;
                Ok(crate::Error::UnificationFailure(lhs, rhs, err))
            })
            .map_err(|e| match e {
                Ok(e) => e,
                Err(e) => e,
            })
    }
    pub fn infer(&mut self, term: TermPtr) -> Result<(TermPtr, ValuePtr)> {
        todo!("infer")
    }
    pub fn check(&mut self, term: TermPtr, ty: ValuePtr) -> Result<TermPtr> {
        todo!("check")
    }

    fn solve_with_partial_renaming(
        &mut self,
        gamma: DBLvl,
        meta: MetaVar,
        (mut renaming, pruning): (PartialRenaming, Option<Pruning>),
        value: ValuePtr,
    ) -> Result<()> {
        trace!(
            "solving meta {meta:?} with right-hand side {:?}",
            quote(gamma, value.clone(), &self.meta)
        );
        let MetaEntry::Unsolved { ty, blocking } = self.meta.get_meta(meta).clone() else {
            return Err(crate::Error::InvalidUnification(Error::SolvedMeta));
        };
        if let Some(pruning) = pruning {
            prune_type(pruning.iter().rev().copied(), ty.clone(), &mut self.meta)?;
        }

        renaming.with_occ(meta, |renaming| {
            let rhs = renaming.rename(value, &mut self.meta)?;
            let rhs = stack_lambdas(renaming.dom, ty.clone(), rhs, &mut self.meta)?;
            let solution = Environment::new().evaluate(rhs, &mut self.meta)?;
            self.meta
                .set_meta(meta, MetaEntry::Solved { val: solution, ty });

            for i in blocking.iter().cloned() {
                self.retry_check(i)?;
            }

            Ok(())
        })
    }

    pub fn solve(
        &mut self,
        gamma: DBLvl,
        meta: MetaVar,
        spine: &Spine,
        rhs: ValuePtr,
    ) -> Result<()> {
        let renaming = PartialRenaming::invert(gamma, spine, &self.meta)?;
        self.solve_with_partial_renaming(gamma, meta, renaming, rhs)
    }
}

#[derive(Default)]
struct PartialRenaming {
    dom: DBLvl,
    cod: DBLvl,
    rename: FxHashMapRand<DBLvl, DBLvl>,
    occ: Option<MetaVar>,
}

impl PartialRenaming {
    fn lift<F, R>(&mut self, f: F) -> R
    where
        F: FnOnce(&mut Self) -> R,
    {
        self.lift_inplace();
        let res = f(self);
        self.dom.0 -= 1;
        self.cod.0 -= 1;
        self.rename.remove(&self.cod);
        res
    }
    fn lift_inplace(&mut self) {
        self.dom.0 += 1;
        self.cod.0 += 1;
        self.rename.insert(self.cod, self.dom);
    }
    fn skip(&mut self) {
        self.cod.0 += 1;
    }
    fn with_occ<F, R>(&mut self, occ: MetaVar, f: F) -> R
    where
        F: FnOnce(&mut Self) -> R,
    {
        let old_occ = self.occ;
        self.occ = Some(occ);
        let res = f(self);
        self.occ = old_occ;
        res
    }
    fn invert(gamma: DBLvl, spine: &Spine, mctx: &MetaContext) -> Result<(Self, Option<Pruning>)> {
        let mut rename = FxHashMapRand::default();
        let mut nonlinear = FxHashSetRand::default();
        let mut dom = DBLvl::zero();
        let mut new_spine = Vec::with_capacity(spine.len());
        for (val, icit) in spine.iter() {
            let val = mctx.force(val.clone())?;
            match val.data() {
                Value::Rigid(level, s) if s.is_empty() => {
                    let rename_entry = rename.entry(*level);
                    let nonlinear_entry = nonlinear.entry(*level);
                    if let HMapEntry::Occupied(x) = rename_entry {
                        x.remove();
                        nonlinear_entry.insert();
                    } else if matches!(nonlinear_entry, HSetEntry::Vacant(_)) {
                        rename_entry.insert_entry(dom);
                    }
                    new_spine.push((*level, *icit));
                }
                _ => {
                    return Err(crate::Error::InvalidUnification(Error::Unsupported));
                }
            }
            dom = dom.next();
        }
        let pruning = if nonlinear.is_empty() {
            None
        } else {
            Some(
                new_spine
                    .into_iter()
                    .map(|(level, icit)| {
                        if nonlinear.contains(&level) {
                            None
                        } else {
                            Some(icit)
                        }
                    })
                    .collect(),
            )
        };
        Ok((
            Self {
                occ: None,
                dom,
                cod: gamma,
                rename,
            },
            pruning,
        ))
    }

    fn rename(&mut self, value: ValuePtr, mctx: &mut MetaContext) -> Result<TermPtr> {
        let value = mctx.force(value)?;

        match value.data() {
            Value::Flex(m, sp) => match &self.occ {
                Some(n) if m == n => Err(crate::Error::InvalidUnification(Error::OccursCheck)),
                _ => self.prune_flex(*m, sp, mctx, value.span),
            },
            Value::Rigid(x, sp) => {
                if let Some(y) = self.rename.get(x) {
                    let idx = y.to_index(self.dom);
                    let var = with_span(Term::Var(idx), value.span);
                    self.rename_spine(var, sp, mctx)
                } else {
                    Err(crate::Error::InvalidUnification(Error::EscapingRigid))
                }
            }
            Value::Lambda(x, icit, closure) => {
                let var = with_span(Value::var(self.cod), x.span);
                let body = closure.apply(var, mctx)?;
                let body = self.lift(|this| this.rename(body, mctx))?;
                Ok(with_span(Term::Lambda(x.clone(), *icit, body), value.span))
            }
            Value::Pi(x, icit, arg_ty, closure) => {
                let arg_ty = self.rename(arg_ty.clone(), mctx)?;
                let var = with_span(Value::var(self.cod), x.span);
                let body = closure.apply(var, mctx)?;
                let body = self.lift(|this| this.rename(body, mctx))?;
                Ok(with_span(
                    Term::Pi(x.clone(), *icit, arg_ty, body),
                    value.span,
                ))
            }
            Value::Universe => Ok(with_span(Term::Universe, value.span)),
        }
    }

    fn rename_spine(
        &mut self,
        term: TermPtr,
        spine: &Spine,
        mctx: &mut MetaContext,
    ) -> Result<TermPtr> {
        spine.iter().try_fold(term, |acc, (val, icit)| {
            let rhs = self.rename(val.clone(), mctx)?;
            let span = acc.span;
            Ok(with_span(Term::App(acc, rhs, *icit), span))
        })
    }

    fn prune_flex(
        &mut self,
        mut meta: MetaVar,
        spine: &Spine,
        mctx: &mut MetaContext,
        span: Span,
    ) -> Result<TermPtr> {
        enum PruneState {
            Renaming,
            NonRenaming,
            NeedsPruning,
        }
        let mut state = PruneState::Renaming;
        let mut sp = Vec::with_capacity(spine.len());
        for (val, icit) in spine.iter() {
            let val = mctx.force(val.clone())?;
            match val.data() {
                Value::Rigid(x, s) if s.is_empty() => {
                    let rename = self.rename.get(x);
                    if let Some(rename) = rename {
                        let idx = rename.to_index(self.dom);
                        let var = with_span(Term::Var(idx), val.span);
                        sp.push(Some((var, *icit)));
                    } else if matches!(state, PruneState::NonRenaming) {
                        return Err(crate::Error::InvalidUnification(Error::Unsupported));
                    } else {
                        trace!("pruning variable");
                        sp.push(None);
                        state = PruneState::NeedsPruning;
                    }
                }
                _ if matches!(state, PruneState::NeedsPruning) => {
                    trace!("cannot prune in non-renaming state");
                    return Err(crate::Error::InvalidUnification(Error::Unsupported));
                }
                _ => {
                    let val = self.rename(val.clone(), mctx)?;
                    sp.push(Some((val, *icit)));
                    state = PruneState::NonRenaming;
                }
            }
        }
        if matches!(state, PruneState::NeedsPruning) {
            meta = prune_meta(sp.iter().map(|v| v.as_ref().map(|(_, i)| *i)), meta, mctx)?;
        } else if matches!(mctx.get_meta(meta), MetaEntry::Solved { .. }) {
            return Err(crate::Error::InvalidUnification(Error::SolvedMeta));
        }
        let term = with_span(Term::Meta(meta), span);
        Ok(sp
            .iter()
            .filter_map(Option::as_ref)
            .rfold(term, |acc, (val, icit)| {
                let val = val.clone();
                let span = acc.span;
                with_span(Term::App(acc, val, *icit), span)
            }))
    }
}

fn stack_lambdas(
    level: DBLvl,
    mut ty: ValuePtr,
    term: TermPtr,
    mctx: &MetaContext,
) -> Result<TermPtr> {
    let mut current = DBLvl::zero();
    let mut result = TERM_PLACEHOLDER.with(|hole| hole.clone());
    let mut cursor = &mut result;
    while current != level {
        current = current.next();
        ty = mctx.force(ty.clone())?;
        if let Value::Pi(x, icit, _, closure) = ty.data() {
            let x = if x.is_anon() {
                WithSpan::new(Ustr::from(&format!("α{}", current.0)), x.span)
            } else {
                x.clone()
            };
            let var = with_span(Value::var(current), x.span);
            let icit = *icit;
            ty = closure.apply(var, mctx)?;
            let hole = TERM_PLACEHOLDER.with(|hole| hole.clone());
            *cursor = with_span(Term::Lambda(x, icit, hole), term.span);
            let Term::Lambda(_, _, body) = Rc::make_mut(cursor).data_mut() else {
                unreachable!("expected lambda type");
            };
            cursor = body;
        } else {
            return Err(crate::Error::InvalidUnification(Error::Unsupported));
        }
        current = current.next();
    }
    *cursor = term;
    Ok(result)
}

fn prune_type<'a>(
    mut pruning: impl Iterator<Item = Option<Icit>>,
    mut ty: ValuePtr,
    mctx: &mut MetaContext,
) -> Result<TermPtr> {
    let mut renaming = PartialRenaming::default();
    let mut result = TERM_PLACEHOLDER.with(|hole| hole.clone());
    let mut cursor = &mut result;
    loop {
        ty = mctx.force(ty)?;
        let Some(mask) = pruning.next() else { break };
        let Value::Pi(x, icit, arg_ty, closure) = ty.data() else {
            return Err(crate::Error::InvalidUnification(Error::Unsupported));
        };
        let var = with_span(Value::var(renaming.cod), x.span);
        if mask.is_none() {
            let arg_ty = renaming.rename(arg_ty.clone(), mctx)?;
            let hole = TERM_PLACEHOLDER.with(|hole| hole.clone());
            *cursor = with_span(Term::Pi(x.clone(), *icit, arg_ty, hole), ty.span);
            let Term::Pi(_, _, _, body) = Rc::make_mut(cursor).data_mut() else {
                unreachable!("expected pi type");
            };
            cursor = body;
            renaming.lift_inplace();
        } else {
            renaming.skip();
        }
        ty = closure.apply(var, mctx)?;
    }
    *cursor = renaming.rename(ty, mctx)?;
    Ok(result)
}

fn prune_meta<'a, I>(pruning: I, meta: MetaVar, mctx: &mut MetaContext) -> Result<MetaVar>
where
    I: Iterator<Item = Option<Icit>> + Clone + ExactSizeIterator + DoubleEndedIterator,
{
    let MetaEntry::Unsolved { ty, blocking } = mctx.get_meta(meta).clone() else {
        return Err(crate::Error::InvalidUnification(Error::SolvedMeta));
    };
    let len = pruning.len();
    let pruned_ty = prune_type(pruning.clone().rev(), ty.clone(), mctx)?;
    let pruned_ty = Environment::new().evaluate(pruned_ty, mctx)?;
    let span = pruned_ty.span;
    let new_meta = mctx.new_meta(pruned_ty, blocking);
    let solution_body = with_span(
        Term::AppPruning(with_span(Term::Meta(new_meta), span), pruning.collect()),
        Span::default(),
    );
    let solution_lvl = DBLvl(len);
    let solution_lambda = stack_lambdas(solution_lvl, ty.clone(), solution_body, mctx)?;
    let solution = Environment::new().evaluate(solution_lambda, mctx)?;
    mctx.set_meta(meta, MetaEntry::Solved { val: solution, ty });
    Ok(new_meta)
}
