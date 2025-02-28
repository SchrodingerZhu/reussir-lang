use std::rc::Rc;

use super::{FieldName, QualifiedName, UniqueName};
use dynforest::{Connection, Handle as ConnHandle};
use rustc_hash::{FxHashMapRand, FxHashSetRand};
use ustr::Ustr;

use crate::syntax::WithSpan;
pub type TermPtr = Rc<WithSpan<Term>>;

#[derive(Clone)]
pub enum Term {
    /// Integer literal
    Integer(rug::Integer),
    /// Float literal
    Float(rug::Float),
    /// String literal
    Str(Ustr),
    /// Boolean literal
    Boolean(bool),
    /// function call
    FuncCall {
        target: WithSpan<QualifiedName>,
        ty_args: Box<[TermPtr]>,
        arguments: Box<[TermPtr]>,
    },
    /// constructor call
    CtorCall {
        target: QualifiedName,
        ty_args: Box<[TermPtr]>,
        arguments: Box<[(FieldName, TermPtr)]>,
    },
    /// closure call
    App(TermPtr, TermPtr),
    /// project a field out of a record
    Proj {
        value: TermPtr,
        field: FieldName,
    },
    /// match variant
    Match {},
    /// cast variant
    Cast {},
    /// lift function into a closure
    FuncAbs {
        target: QualifiedName,
        ty_args: Box<[TermPtr]>,
    },
    /// lift constructor into a closure
    CtorAbs {
        target: QualifiedName,
        ty_args: Box<[TermPtr]>,
    },
    /// Lambda expression
    Lambda {
        binding: UniqueName,
        body: TermPtr,
    },
    /// Let binding
    Let {
        name: UniqueName,
        binding: TermPtr,
        body: TermPtr,
    },
    /// Sequence
    Seq(TermPtr, TermPtr),
    /// Builtin Integer types,
    IntTy(crate::syntax::r#type::Int),
    /// Builtin Float types
    FloatTy(crate::syntax::r#type::Float),
    /// Unit Type,
    UnitTy,
    /// Never Type,
    NeverTy,
    /// Pi type
    Pi {
        name: UniqueName,
        arg: TermPtr,
        body: TermPtr,
    },
    Var(UniqueName),
    StrTy,
    BooleanTy,
    Universe,
    MetaVar(UniqueName),
    CheckVar,
    Invalid,
}

#[derive(Default)]
struct Unifier<'a>(FxHashMapRand<&'a UniqueName, ConnHandle>);

impl<'a> Unifier<'a> {
    fn new() -> Self {
        Self::default()
    }

    fn get_handle(&mut self, target: &'a UniqueName) -> ConnHandle {
        self.0.entry(target).or_default().clone()
    }

    fn with<R, F: for<'x> FnOnce(&'x mut Self) -> R>(
        &mut self,
        a: &'a UniqueName,
        b: &'a UniqueName,
        conti: F,
    ) -> R {
        let h0 = self.get_handle(a);
        let h1 = self.get_handle(b);
        let conn = h0.connect(&h1);
        debug_assert!(conn.is_some());

        conti(self)
    }

    fn is_equivalent(&mut self, a: &'a UniqueName, b: &'a UniqueName) -> bool {
        let h0 = self.get_handle(a);
        let h1 = self.get_handle(b);
        h0.is_connected(&h1)
    }
}

impl Term {
    /// For alpha equivalence, unifier is simply a hashmap tracks variable mapping
    fn alpha_equivalence_impl<'a, 'b>(&'a self, other: &'a Self, unifier: &mut Unifier<'b>) -> bool
    where
        'a: 'b,
    {
        match (self, other) {
            (Term::Var(a), Term::Var(b)) => unifier.is_equivalent(a, b),
            (Term::Integer(a), Term::Integer(b)) => a == b,
            (
                Term::Lambda {
                    binding: va,
                    body: ba,
                },
                Term::Lambda {
                    binding: vb,
                    body: bb,
                },
            ) => unifier.with(va, vb, |unifier| ba.alpha_equivalence_impl(bb, unifier)),
            (
                Term::Pi {
                    name: na,
                    arg: aa,
                    body: ba,
                },
                Term::Pi {
                    name: nb,
                    arg: ab,
                    body: bb,
                },
            ) => {
                aa.alpha_equivalence_impl(ab, unifier)
                    && unifier.with(na, nb, |unifier| ba.alpha_equivalence_impl(bb, unifier))
            }
            _ if std::mem::discriminant(self) == std::mem::discriminant(other) => todo!(),
            _ => false,
        }
    }

    pub fn is_alpha_equivalent(&self, other: &Self) -> bool {
        self.alpha_equivalence_impl(other, &mut Unifier::new())
    }
}

impl std::fmt::Display for Term {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Term::Integer(x) => write!(f, "{}", x),
            Term::Float(x) => write!(f, "{}", x),
            Term::Str(x) => write!(f, "{x:?}"),
            Term::Boolean(x) => write!(f, "{x}"),
            Term::FuncCall {
                target,
                ty_args,
                arguments,
            } => todo!(),
            Term::CtorCall {
                target,
                ty_args,
                arguments,
            } => todo!(),
            Term::App(gc, gc1) => {
                write!(f, "({} {})", ***gc, ***gc1)
            }
            Term::Proj { value, field } => todo!(),
            Term::Match {} => todo!(),
            Term::Cast {} => todo!(),
            Term::FuncAbs { target, ty_args } => todo!(),
            Term::CtorAbs { target, ty_args } => todo!(),
            Term::Lambda { binding, body } => {
                write!(f, "λ{}.{}", **binding.0, body.0)
            }
            Term::Let {
                name,
                binding,
                body,
            } => todo!(),
            Term::Seq(gc, gc1) => todo!(),
            Term::IntTy(_) => todo!(),
            Term::FloatTy(_) => todo!(),
            Term::Pi { name, arg, body } => todo!(),
            Term::Var(unique_name) => {
                write!(f, "{}", unique_name.0.0)
            }
            Term::StrTy => todo!(),
            Term::BooleanTy => todo!(),
            Term::Universe => todo!(),
            Term::MetaVar(unique_name) => todo!(),
            Term::CheckVar => todo!(),
            Term::Invalid => write!(f, "<invalid>"),
            Term::UnitTy => write!(f, "()"),
            Term::NeverTy => write!(f, "!"),
        }
    }
}
