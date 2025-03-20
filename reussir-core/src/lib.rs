use std::{backtrace::Backtrace, borrow::Cow};

use meta::MetaVar;
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
