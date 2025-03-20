use std::cell::RefCell;

use thiserror::Error;
use tracing::trace;

use crate::{
    Result,
    ctx::Context,
    eval::Environment,
    meta::{CheckVar, MetaContext, MetaEntry, MetaVar},
    term::TermPtr,
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

impl Elaborator {
    pub fn new() -> Self {
        Self {
            ctx: Context::new(),
            meta: MetaContext::new(),
        }
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
                todo!()
            }
        }
        Ok(())
    }
    fn retry_check(&mut self, var: CheckVar) -> Result<()> {
        todo!("retry check")
    }
}
