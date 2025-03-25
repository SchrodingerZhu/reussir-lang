use std::rc::Rc;

use either::Either;

use crate::{
    meta::{CheckVar, MetaVar},
    utils::{DBIdx, Icit, Name, Pruning, WithSpan},
};

pub type TermPtr = Rc<WithSpan<Term>>;

#[derive(Debug, Clone)]
pub enum Term {
    Hole,
    Var(DBIdx),
    Lambda(Name, Either<Icit, Name>, Option<TermPtr>, TermPtr),
    App(TermPtr, TermPtr, Either<Icit, Name>),
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
