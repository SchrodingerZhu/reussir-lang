use std::rc::Rc;

use either::Either;

use crate::utils::{Icit, Name, Pruning, WithSpan};

pub type SurfPtr = Rc<WithSpan<Surface>>;

#[derive(Debug, Clone)]
pub enum Surface {
    Hole,
    Var(Name),
    Lambda(Name, Either<Icit, Name>, Option<SurfPtr>, SurfPtr),
    App(SurfPtr, SurfPtr, Either<Icit, Name>),
    AppPruning(SurfPtr, Pruning),
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
    use crate::utils::{Span, with_span};
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

    pub(crate) fn pi<F>(name: &str, implicit: bool, arg: SurfPtr, body: F) -> SurfPtr
    where
        F: FnOnce(SurfPtr) -> SurfPtr,
    {
        let name = Name::new(name.into(), Default::default());
        let var = with_default_span(Surface::Var(name.clone()));
        let body = body(var);
        with_default_span(Surface::Pi(name, Icit::Impl, arg, body))
    }

    pub(crate) fn arrow(x: SurfPtr, y: SurfPtr) -> SurfPtr {
        let name = Name::new("".into(), Default::default());
        let var = with_default_span(Surface::Var(name.clone()));
        with_default_span(Surface::Pi(name, Icit::Impl, x, y))
    }

    pub(crate) fn app<const N: usize>(f: SurfPtr, x: [SurfPtr; N]) -> SurfPtr {
        x.into_iter().fold(f, |f, x| {
            with_default_span(Surface::App(f, x, Left(Icit::Impl)))
        })
    }
}
