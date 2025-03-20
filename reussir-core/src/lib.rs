use std::{backtrace::Backtrace, borrow::Cow};

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
}

impl Error {
    pub fn internal<E: Into<Cow<'static, str>>>(error: E) -> Self {
        Self::Internal(error.into(), Box::new(Backtrace::capture()))
    }
    pub fn unresolved_variable(name: UniqueName) -> Self {
        Self::UnresolvedVariable(name)
    }
}

pub type Result<T> = std::result::Result<T, Error>;
