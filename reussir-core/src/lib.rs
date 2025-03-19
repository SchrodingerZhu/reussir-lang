use std::backtrace::Backtrace;

use thiserror::Error;

mod ctx;
mod elab;
mod eval;
mod meta;
mod term;
mod utils;
mod value;

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
