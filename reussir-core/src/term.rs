use std::rc::Rc;

use either::Either;

use crate::{
    meta::{CheckVar, MetaVar},
    utils::{DBIdx, Icit, Name, Pruning, WithSpan},
};

pub type TermPtr = Rc<WithSpan<Term>>;

#[derive(Debug, Clone)]
pub enum Term {
    Var(DBIdx),
    Lambda(Name, Icit, TermPtr),
    App(TermPtr, TermPtr, Icit),
    AppPruning(TermPtr, Pruning),
    Universe,
    Pi(Name, Icit, TermPtr, TermPtr),
    Let {
        name: Name,
        ty: TermPtr,
        term: TermPtr,
        body: TermPtr,
    },
    Meta(MetaVar),
    Postponed(CheckVar),
}
