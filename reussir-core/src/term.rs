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
    Pi(Option<UniqueName>, Icit, TermPtr, TermPtr),
    Let(UniqueName, TermPtr, TermPtr, TermPtr),
    Meta(MetaVar),
    Postponed(CheckVar),
}
