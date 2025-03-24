#![feature(new_range_api)]
#![feature(hash_set_entry)]
#![feature(unique_rc_arc)]
use term::TermPtr;
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
    #[error("failed to unify {0:?} with {1:?} ({2})")]
    UnificationFailure(TermPtr, TermPtr, elab::Error),
    #[error("icitness mismatch: expected {0}, got {1}")]
    IcitMismatch(utils::Icit, utils::Icit),
    #[error("invalid unification pattern {0}")]
    InvalidUnification(elab::Error),
}

pub type Result<T> = std::result::Result<T, Error>;
