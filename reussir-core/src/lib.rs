use std::backtrace::Backtrace;

use thiserror::Error;

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
    Internal(String, Box<Backtrace>),
}

impl Error {
    pub fn internal<E: std::fmt::Display>(error: E) -> Self {
        Self::Internal(error.to_string(), Box::new(Backtrace::capture()))
    }
}

pub type Result<T> = std::result::Result<T, Error>;
