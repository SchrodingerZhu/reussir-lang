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

use std::collections::hash_set::Entry as HSetEntry;
use std::{collections::hash_map::Entry as HMapEntry, rc::Rc};

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
    pub fn unify_impl(&mut self, lhs: ValuePtr, rhs: ValuePtr) -> Option<()> {
        todo!("unify")
    }
    pub fn unify(&mut self, lhs: ValuePtr, rhs: ValuePtr, err: Error) -> Result<()> {
        self.unify_impl(lhs.clone(), rhs.clone())
            .ok_or_else(|| {
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
    fn invert(gamma: DBLvl, spine: &Spine, mctx: &MetaContext) -> Option<(Self, Option<Pruning>)> {
        let mut rename = FxHashMapRand::default();
        let mut nonlinear = FxHashSetRand::default();
        let mut dom = DBLvl::zero();
        let mut new_spine = Vec::with_capacity(spine.len());
        for (val, icit) in spine.iter() {
            let val = mctx
                .force(val.clone())
                .inspect_err(|e| error!("failed to force spine value: {e}"))
                .ok()?;
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
                    trace!("invalid unification pattern");
                    return None;
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
        Some((
            Self {
                occ: None,
                dom,
                cod: gamma,
                rename,
            },
            pruning,
        ))
    }

    fn rename(&mut self, value: ValuePtr, mctx: &MetaContext) -> Option<TermPtr> {
        let value = mctx
            .force(value)
            .inspect_err(|e| error!("failed to force value: {e}"))
            .ok()?;

        match value.data() {
            Value::Flex(m, sp) => match &self.occ {
                Some(n) if m == n => {
                    trace!("occurs check failed during renaming");
                    None
                }
                _ => self.prune_flex(*m, sp, mctx),
            },
            Value::Rigid(x, sp) => {
                if let Some(y) = self.rename.get(x) {
                    let idx = y.to_index(self.dom);
                    let var = with_span(Term::Var(idx), value.span);
                    self.rename_spine(var, sp, mctx)
                } else {
                    trace!("encountered escaping rigid variable during renaming");
                    return None;
                }
            }
            Value::Lambda(x, icit, closure) => {
                let var = with_span(Value::var(self.cod), x.span);
                let body = closure
                    .apply(var, mctx)
                    .inspect_err(|e| {
                        trace!("failed to apply closure: {e}");
                    })
                    .ok()?;
                let body = self.lift(|this| this.rename(body, mctx))?;
                Some(with_span(Term::Lambda(x.clone(), *icit, body), value.span))
            }
            Value::Pi(x, icit, arg_ty, closure) => {
                let arg_ty = self.rename(arg_ty.clone(), mctx)?;
                let var = with_span(Value::var(self.cod), x.span);
                let body = closure
                    .apply(var, mctx)
                    .inspect_err(|e| {
                        trace!("failed to apply closure: {e}");
                    })
                    .ok()?;
                let body = self.lift(|this| this.rename(body, mctx))?;
                Some(with_span(
                    Term::Pi(x.clone(), *icit, arg_ty, body),
                    value.span,
                ))
            }
            Value::Universe => Some(with_span(Term::Universe, value.span)),
        }
    }

    fn rename_spine(
        &mut self,
        term: TermPtr,
        spine: &Spine,
        mctx: &MetaContext,
    ) -> Option<TermPtr> {
        spine.iter().try_fold(term, |acc, (val, icit)| {
            let rhs = self.rename(val.clone(), mctx)?;
            let span = acc.span;
            Some(with_span(Term::App(acc, rhs, *icit), span))
        })
    }

    fn prune_flex(&mut self, meta: MetaVar, spine: &Spine, mctx: &MetaContext) -> Option<TermPtr> {
        todo!("prune_flex")
    }

    fn prune_type<'a>(
        mut pruning: impl Iterator<Item = &'a Option<Icit>>,
        mut ty: ValuePtr,
        mctx: &MetaContext,
    ) -> Option<TermPtr> {
        let mut renaming = PartialRenaming::default();
        let mut result = TERM_PLACEHOLDER.with(|hole| hole.clone());
        let mut cursor = &mut result;
        loop {
            ty = mctx
                .force(ty)
                .inspect_err(|e| error!("failed to force type: {e}"))
                .ok()?;
            let Some(mask) = pruning.next() else { break };
            let Value::Pi(x, icit, arg_ty, closure) = ty.data() else {
                error!("prune_type applied to non-pi type");
                return None;
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
            ty = closure
                .apply(var, mctx)
                .inspect_err(|e| {
                    trace!("failed to apply closure: {e}");
                })
                .ok()?;
        }
        *cursor = renaming.rename(ty, mctx)?;
        Some(result)
    }

    fn prune_meta(pruning: &Pruning, meta: MetaVar, mctx: &mut MetaContext) -> Option<MetaVar> {
        let MetaEntry::Unsolved { ty, blocking } = mctx.get_meta(meta) else {
            error!("prune_meta applied to solved meta");
            return None;
        };
        let pruned_ty = Self::prune_type(pruning.iter().rev(), ty.clone(), mctx)?;
        let pruned_ty = Environment::new()
            .evaluate(pruned_ty, mctx)
            .inspect_err(|e| {
                error!("failed to evaluate pruned type: {e}");
            })
            .ok()?;
        let new_meta = mctx.new_meta(pruned_ty, blocking.clone());
        todo!("prune_meta")
    }
}

fn stack_lambdas(
    level: DBLvl,
    mut ty: ValuePtr,
    term: TermPtr,
    mctx: &MetaContext,
) -> Option<TermPtr> {
    let mut current = DBLvl::zero();
    let mut result = TERM_PLACEHOLDER.with(|hole| hole.clone());
    let mut cursor = &mut result;
    while current != level {
        current = current.next();
        ty = mctx
            .force(ty.clone())
            .inspect_err(|e| error!("failed to force type: {e}"))
            .ok()?;
        if let Value::Pi(x, icit, _, closure) = ty.data() {
            let x = if x.is_anon() {
                WithSpan::new(Ustr::from(&format!("α{}", current.0)), x.span)
            } else {
                x.clone()
            };
            let var = with_span(Value::var(current), x.span);
            let icit = *icit;
            ty = closure
                .apply(var, mctx)
                .inspect_err(|e| {
                    trace!("failed to apply closure: {e}");
                })
                .ok()?;
            let hole = TERM_PLACEHOLDER.with(|hole| hole.clone());
            *cursor = with_span(Term::Lambda(x, icit, hole), term.span);
            let Term::Lambda(_, _, body) = Rc::make_mut(cursor).data_mut() else {
                unreachable!("expected lambda type");
            };
            cursor = body;
        } else {
            error!("stack_lambdas applied to non-pi type");
            return None;
        }
        current = current.next();
    }
    *cursor = term;
    Some(result)
}
