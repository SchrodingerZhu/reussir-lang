use std::{backtrace::Backtrace, borrow::Cow};

use meta::MetaVar;
use term::TermPtr;
use thiserror::Error;
use utils::UniqueName;

pub mod ctx;
pub mod elab;
pub mod eval;
pub mod meta;
pub mod term;
pub mod utils;
pub mod value;

#[derive(Error, Debug)]
pub enum Error {
    #[error("{0}")]
    Internal(Cow<'static, str>, Box<Backtrace>),
    #[error("failed to resolve variable {0} within context")]
    UnresolvedVariable(UniqueName),
    #[error("failed to resolve meta {0:?} within context")]
    UnresolvedMeta(MetaVar),
    #[error("failed to unify {0:?} with {1:?} ({2})")]
    UnificationFailure(TermPtr, TermPtr, elab::Error),
    #[error("icitness mismatch: expected {0}, got {1}")]
    IcitMismatch(utils::Icit, utils::Icit),
}

impl Error {
    pub fn internal<E: Into<Cow<'static, str>>>(error: E) -> Self {
        Self::Internal(error.into(), Box::new(Backtrace::capture()))
    }
    pub fn unresolved_variable(name: UniqueName) -> Self {
        Self::UnresolvedVariable(name)
    }
    pub fn unresolved_meta(meta: MetaVar) -> Self {
        Self::UnresolvedMeta(meta)
    }
}

pub type Result<T> = std::result::Result<T, Error>;
