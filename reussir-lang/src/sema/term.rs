use std::rc::Rc;

use super::{FieldName, QualifiedName, UniqueName};
use dynforest::{Connection, Handle as ConnHandle};
use rpds::Queue;
use rustc_hash::{FxHashMapRand, FxHashSetRand};
use ustr::Ustr;

use crate::syntax::WithSpan;
pub type TermPtr = Rc<WithSpan<Term>>;

#[derive(Debug, Clone)]
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
    App(
        TermPtr,
        TermPtr,
        /// implicit
        bool,
    ),
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
        implicit: bool,
    },
    /// Let binding
    Let {
        name: UniqueName,
        ty: Option<TermPtr>,
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
        implicit: bool,
    },
    Var(UniqueName),
    StrTy,
    BooleanTy,
    Universe,
    // A hole whose meta variable is not yet assigned
    Hole,
    Meta(usize),
    InsertedMeta(usize, Queue<UniqueName>),
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
                    implicit: i0,
                },
                Term::Lambda {
                    binding: vb,
                    body: bb,
                    implicit: i1,
                },
            ) => i0 == i1 && unifier.with(va, vb, |unifier| ba.alpha_equivalence_impl(bb, unifier)),
            (
                Term::Pi {
                    name: na,
                    arg: aa,
                    body: ba,
                    implicit: i0,
                },
                Term::Pi {
                    name: nb,
                    arg: ab,
                    body: bb,
                    implicit: i1,
                },
            ) => {
                i0 == i1
                    && aa.alpha_equivalence_impl(ab, unifier)
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
        stacker::maybe_grow(32 * 1024, 1024 * 1024, || match self {
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
            Term::App(gc, gc1, implicit) => {
                if !implicit {
                    write!(f, "({} {})", ***gc, ***gc1)
                } else {
                    write!(f, "({} {{{}}})", ***gc, ***gc1)
                }
            }
            Term::Proj { value, field } => todo!(),
            Term::Match {} => todo!(),
            Term::Cast {} => todo!(),
            Term::FuncAbs { target, ty_args } => todo!(),
            Term::CtorAbs { target, ty_args } => todo!(),
            Term::Lambda {
                binding,
                body,
                implicit,
            } => {
                if !implicit {
                    write!(f, "λ{}.{}", **binding.0, body.0)
                } else {
                    write!(f, "λ{{{}}}.{}", **binding.0, body.0)
                }
            }
            Term::Let {
                name,
                ty,
                binding,
                body,
            } => {
                if let Some(ty) = ty {
                    write!(
                        f,
                        "let {} : {} = {} in {}",
                        **name.0, ty.0, binding.0, body.0
                    )
                } else {
                    write!(f, "let {} = {} in {}", **name.0, binding.0, body.0)
                }
            }
            Term::Seq(gc, gc1) => todo!(),
            Term::IntTy(_) => todo!(),
            Term::FloatTy(_) => todo!(),
            Term::Pi {
                name,
                arg,
                body,
                implicit,
            } => {
                if (**name.0).is_empty() {
                    write!(f, "{} -> {}", arg.0, body.0)
                } else if !implicit {
                    write!(f, "Π({} : {}).{}", **name.0, arg.0, body.0)
                } else {
                    write!(f, "Π{{{} : {}}}.{}", **name.0, arg.0, body.0)
                }
            }
            Term::Var(unique_name) => {
                write!(f, "{}", unique_name.0.0)
            }
            Term::StrTy => todo!(),
            Term::BooleanTy => todo!(),
            Term::Universe => write!(f, "U"),
            Term::Meta(x) | Term::InsertedMeta(x, _) => write!(f, "?{x}"),
            Term::CheckVar => todo!(),
            Term::Invalid => write!(f, "<invalid>"),
            Term::UnitTy => write!(f, "()"),
            Term::NeverTy => write!(f, "!"),
            Term::Hole => write!(f, "_"),
        })
    }
}

#[cfg(test)]
pub(crate) mod test {
    use chumsky::span::SimpleSpan;

    use crate::sema::eval::evaluate;

    use super::*;

    pub(crate) fn lam<F, const N: usize>(name: [&str; N], body: F) -> TermPtr
    where
        F: FnOnce([TermPtr; N]) -> TermPtr,
    {
        let fake_span = SimpleSpan::new(0, 0);
        let names = name.map(|name| UniqueName::new(name, fake_span));
        let vars = names
            .clone()
            .map(|name| Rc::new(WithSpan(Term::Var(name), fake_span)));
        let body = body(vars);
        names.into_iter().rev().fold(body, |body, binding| {
            Rc::new(WithSpan(
                Term::Lambda {
                    binding,
                    body,
                    implicit: false,
                },
                fake_span,
            ))
        })
    }

    pub(crate) fn universe() -> TermPtr {
        Rc::new(WithSpan(Term::Universe, SimpleSpan::new(0, 0)))
    }

    pub(crate) fn hole() -> TermPtr {
        Rc::new(WithSpan(Term::Hole, SimpleSpan::new(0, 0)))
    }

    pub(crate) fn r#let<F>(name: &str, binding: TermPtr, ty: TermPtr, b: F) -> TermPtr
    where
        F: FnOnce(TermPtr) -> TermPtr,
    {
        let fake_span = SimpleSpan::new(0, 0);
        let name = UniqueName::new(name, fake_span);
        let var = Rc::new(WithSpan(Term::Var(name.clone()), fake_span));
        let body = b(var);
        let ty = Some(ty);
        Rc::new(WithSpan(
            Term::Let {
                name,
                ty,
                binding,
                body,
            },
            fake_span,
        ))
    }

    pub(crate) fn pi<F>(name: &str, implicit: bool, arg: TermPtr, body: F) -> TermPtr
    where
        F: FnOnce(TermPtr) -> TermPtr,
    {
        let fake_span = SimpleSpan::new(0, 0);
        let name = UniqueName::new(name, fake_span);
        let var = Rc::new(WithSpan(Term::Var(name.clone()), fake_span));
        let body = body(var);
        Rc::new(WithSpan(
            Term::Pi {
                name,
                body,
                arg,
                implicit,
            },
            fake_span,
        ))
    }

    pub(crate) fn arrow(x: TermPtr, y: TermPtr) -> TermPtr {
        let fake_span = SimpleSpan::new(0, 0);
        let name = UniqueName::new("", fake_span);
        let var = Rc::new(WithSpan(Term::Var(name.clone()), fake_span));
        Rc::new(WithSpan(
            Term::Pi {
                name,
                body: y,
                arg: x,
                implicit: false,
            },
            fake_span,
        ))
    }

    pub(crate) fn app<const N: usize>(f: TermPtr, x: [TermPtr; N]) -> TermPtr {
        let fake_span = SimpleSpan::new(0, 0);
        x.into_iter().fold(f, |f, x| {
            Rc::new(WithSpan(Term::App(f, x, false), fake_span))
        })
    }
}
