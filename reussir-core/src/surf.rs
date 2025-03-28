use std::rc::Rc;

use either::{Either, Left, Right};

use crate::utils::Icit::Impl;
use crate::utils::{Icit, Name, Pruning, WithSpan};

pub type SurfPtr = Rc<WithSpan<Surface>>;

#[derive(Debug, Clone)]
pub enum Surface {
    Hole,
    Var(Name),
    Lambda(Name, Either<Icit, Name>, Option<SurfPtr>, SurfPtr),
    App(SurfPtr, SurfPtr, Either<Icit, Name>),
    Universe,
    Pi(Name, Icit, SurfPtr, SurfPtr),
    Let {
        name: Name,
        ty: SurfPtr,
        term: SurfPtr,
        body: SurfPtr,
    },
}

#[cfg(test)]
pub(crate) mod test {
    use super::*;
    use crate::term::{Term, TermPtr};
    use crate::utils::{DBIdx, Span, with_span};
    use crate::value::{Value, ValuePtr};
    use either::Left;
    use ustr::Ustr;

    pub(crate) fn with_default_span<T>(t: T) -> Rc<WithSpan<T>> {
        Rc::new(WithSpan::new(t, Default::default()))
    }

    pub(crate) fn lam<F, const N: usize>(name: [&str; N], body: F) -> SurfPtr
    where
        F: FnOnce([SurfPtr; N]) -> SurfPtr,
    {
        let names = name
            .map(Ustr::from)
            .map(|name| Name::new(name, Default::default()));
        let vars = names
            .clone()
            .map(|name| with_default_span(Surface::Var(name)));
        let body = body(vars);
        names.into_iter().rev().fold(body, |body, name| {
            with_default_span(Surface::Lambda(name, Left(Icit::Expl), None, body))
        })
    }

    pub(crate) fn universe() -> SurfPtr {
        with_default_span(Surface::Universe)
    }

    pub(crate) fn term_universe() -> TermPtr {
        with_default_span(Term::Universe)
    }

    pub(crate) fn hole() -> SurfPtr {
        with_default_span(Surface::Hole)
    }

    pub(crate) fn r#let<F>(name: &str, term: SurfPtr, ty: SurfPtr, b: F) -> SurfPtr
    where
        F: FnOnce(SurfPtr) -> SurfPtr,
    {
        let name = Name::new(name.into(), Default::default());
        let var = with_default_span(Surface::Var(name.clone()));
        let body = b(var);
        with_default_span(Surface::Let {
            name,
            ty,
            term,
            body,
        })
    }

    pub(crate) fn pi<F>(name: &str, icit: Icit, arg: SurfPtr, body: F) -> SurfPtr
    where
        F: FnOnce(SurfPtr) -> SurfPtr,
    {
        let name = Name::new(name.into(), Default::default());
        let var = with_default_span(Surface::Var(name.clone()));
        let body = body(var);
        with_default_span(Surface::Pi(name, icit, arg, body))
    }

    pub(crate) fn term_pi(name: &str, icit: Icit, arg: TermPtr, body: TermPtr) -> TermPtr {
        let name = Name::new(name.into(), Default::default());
        let var = with_default_span(Term::Var(DBIdx(0)));
        with_default_span(Term::Pi(name, icit, arg, body))
    }

    pub(crate) fn term_var(idx: usize) -> TermPtr {
        with_default_span(Term::Var(DBIdx(idx)))
    }

    pub(crate) fn arrow(x: SurfPtr, y: SurfPtr) -> SurfPtr {
        let name = Name::new("".into(), Default::default());
        let var = with_default_span(Surface::Var(name.clone()));
        with_default_span(Surface::Pi(name, Icit::Expl, x, y))
    }

    pub(crate) fn app<const N: usize>(f: SurfPtr, x: [SurfPtr; N]) -> SurfPtr {
        x.into_iter().fold(f, |f, x| {
            with_default_span(Surface::App(f, x, Left(Icit::Expl)))
        })
    }
}

impl std::fmt::Display for Surface {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Surface::Hole => write!(f, "_"),
            Surface::Var(x) => write!(f, "{}", x.data()),
            Surface::Lambda(x, Left(Icit::Expl), ty, body) => {
                write!(f, "\\{}", x.data())?;
                if let Some(ty) = ty {
                    write!(f, ": {}", ty.data())?;
                }
                write!(f, ".")?;
                body.fmt(f)
            }
            Surface::Lambda(x, Left(Icit::Impl), ty, body) => {
                write!(f, "\\{{{}", x.data())?;
                if let Some(ty) = ty {
                    write!(f, ": {}", ty.data())?;
                }
                write!(f, "}}.")?;
                body.fmt(f)
            }
            Surface::Lambda(x, Right(n), ty, body) => {
                write!(f, "\\{{{} = {}", x.data(), n.data())?;
                if let Some(ty) = ty {
                    write!(f, ": {}", ty.data())?;
                }
                write!(f, "}}.")?;
                body.fmt(f)
            }
            Surface::App(func, x, Left(Icit::Expl)) => {
                write!(f, "({} {})", func.data(), x.data())
            }
            Surface::App(func, x, Left(Icit::Impl)) => {
                write!(f, "{{{} {}}}", func.data(), x.data())
            }
            Surface::App(func, x, Right(n)) => {
                write!(f, "{{{} {}={}}}", func.data(), n.data(), x.data())
            }
            Surface::Universe => write!(f, "𝓤"),
            Surface::Pi(x, Icit::Expl, ty, body) => {
                write!(f, "Π{}: ", x.data())?;
                ty.fmt(f)?;
                write!(f, ".")?;
                body.fmt(f)
            }
            Surface::Pi(x, Icit::Impl, ty, body) => {
                write!(f, "Π{{{}: ", x.data())?;
                ty.fmt(f)?;
                write!(f, "}}.")?;
                body.fmt(f)
            }
            Surface::Let {
                name,
                ty,
                term,
                body,
            } => {
                write!(f, "let {}: ", name.data())?;
                ty.fmt(f)?;
                write!(f, " = ")?;
                term.fmt(f)?;
                write!(f, ".")?;
                body.fmt(f)
            }
        }
    }
}
