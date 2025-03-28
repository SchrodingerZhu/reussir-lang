use std::rc::Rc;

use crate::{
    meta::{CheckVar, MetaVar},
    utils::{DBIdx, Icit, Name, Pruning, WithSpan},
};
use crate::utils::DBLvl;

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

// impl Term {
//     fn fmt(&self, f: &mut std::fmt::Formatter<'_>, level: DBLvl) -> std::fmt::Result  {
//         match self {
//             Term::Var(idx) => write!(f, "x{}", idx.to_index(level)),
//             Term::Lambda(name, icit, body) => {
//                 if icit == &Icit::Impl {
//                     write!(f, "\\{{{}}}.{}", name, body)
//                 } else {
//                     write!(f, "\\{}.{}", name, body)
//                 }
//             }
//             Term::App(lhs, rhs, icit) => {
//                 if let Icit::Impl = *icit {
//                     write!(f, "({} {{{}}})", lhs, rhs)
//                 } else {
//                     write!(f, "({} {})", rhs, lhs)
//                 }
//             }
//             Term::AppPruning(lhs, pruning) => {
//                 write!(f, "({} {})", lhs, pruning)
//             }
//         }
//     }
// }