use std::rc::Rc;

use super::{FieldName, QualifiedName, UniqueName};
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
        }
    }
}
