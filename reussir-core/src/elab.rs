use rustc_hash::FxHashMapRand;
use thiserror::Error;
use tracing::trace;

use crate::{
    Result,
    ctx::Context,
    eval::{Environment, quote},
    meta::{CheckEntry, CheckVar, MetaContext, MetaEntry, MetaVar},
    term::{Term, TermPtr},
    utils::{Icit, UniqueName, with_span},
    value::{Value, ValuePtr},
};

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
        match self.meta.get_meta(mvar)? {
            MetaEntry::Unsolved { ty, blocking } => {
                trace!(
                    "unifying unconstrained meta {:?} with placeholder {:?}",
                    mvar, expected
                );
                let span = expected.span;
                let solution = self.ctx.close_term(expected, span);
                let solution = Environment::new().evaluate(solution, &self.meta)?;
                let ty = ty.clone();
                let blocking = blocking.clone();
                self.meta
                    .set_meta(mvar, MetaEntry::Solved { val: solution, ty })?;
                for i in blocking.iter().map(CheckVar) {
                    self.retry_check(i)?;
                }
            }
            MetaEntry::Solved { val, .. } => {
                trace!(
                    "unifying solved meta {:?} with placeholder {:?}",
                    mvar, expected
                );
                let lhs = self.ctx.env_mut().evaluate(expected, &self.meta)?;
                let (env, pruning) = self.ctx.env_mut_pruning();
                let meta = &self.meta;
                let rhs = env.app_pruning(val.clone(), pruning, meta, val.span)?;
                self.unify(lhs, rhs)?;
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
        } = self.meta.get_check(var)?
        {
            let ty = self.meta.force(ty)?;
            if let Value::Flex(m, _) = ty.data() {
                trace!("delayed check {var:?} is still blocked by meta {m:?}");
                self.meta.add_blocker(var, *m)?;
            } else {
                trace!("delayed check {var:?} is no longer blocked");
                self.with_context(ctx, |this| {
                    let term = this.check(term, ty)?;
                    this.unify_placeholder(term.clone(), meta)?;
                    this.meta.set_check(var, CheckEntry::Checked(term))?;
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
            } = self.meta.get_check(var)?
            else {
                continue;
            };
            self.with_context(ctx, |this| {
                let inferred = this.infer(term.clone())?;
                let (term, ty) = this.apply_implicit_if_neutral(inferred)?;
                this.meta
                    .set_check(var, CheckEntry::Checked(term.clone()))?;
                this.unify(expected, ty)?;
                this.unify_placeholder(term, meta)
            })?;
        }
        Ok(())
    }
    fn evaluated_close_type(&self, value: ValuePtr, span: (usize, usize)) -> Result<ValuePtr> {
        let term = self.ctx.close_type(quote(value, &self.meta)?, span);
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
                Value::Pi(name, Icit::Impl, arg_ty, body) => {
                    let meta = self.fresh_meta(arg_ty.clone())?;
                    let value = self.ctx.env_mut().evaluate(meta.clone(), &self.meta)?;
                    let span = term.span;
                    term = with_span(Term::App(term, meta, Icit::Impl), span);
                    ty = body.apply(name.clone(), value, &self.meta)?;
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
    pub fn unify(&mut self, lhs: ValuePtr, rhs: ValuePtr) -> Result<()> {
        todo!("unify")
    }
    pub fn infer(&mut self, term: TermPtr) -> Result<(TermPtr, ValuePtr)> {
        todo!("infer")
    }
    pub fn check(&mut self, term: TermPtr, ty: ValuePtr) -> Result<TermPtr> {
        todo!("check")
    }
}

struct PartialRenaming {
    map: FxHashMapRand<UniqueName, UniqueName>,
    inversion: Vec<(UniqueName, Icit)>,
    occ: Option<MetaVar>,
}
impl PartialRenaming {
    fn locally<F, R>(&mut self, name: UniqueName, f: F) -> R
    where
        F: for<'a> FnOnce(&'a mut Self, UniqueName) -> R,
    {
        let fresh = name.refresh();
        self.map.insert(name.clone(), fresh.clone());
        let res = f(self, fresh);
        self.map.remove(&name);
        res
    }
}
