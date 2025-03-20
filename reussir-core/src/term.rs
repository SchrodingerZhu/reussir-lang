use std::rc::Rc;

use crate::{
    meta::{CheckVar, MetaVar},
    utils::{Icit, Pruning, UniqueName, WithSpan},
};

pub type TermPtr = Rc<WithSpan<Term>>;

#[derive(Debug, Clone)]
pub enum Term {
    Hole,
    Var(UniqueName),
    Lambda(UniqueName, Icit, TermPtr),
    App(TermPtr, TermPtr, Icit),
    AppPruning(TermPtr, Pruning),
    Universe,
    Pi(UniqueName, Icit, TermPtr, TermPtr),
    Let {
        name: UniqueName,
        ty: TermPtr,
        term: TermPtr,
        body: TermPtr,
    },
    Meta(MetaVar),
    Postponed(CheckVar),
}
