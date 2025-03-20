use std::{cell::RefCell, fmt::Display, ops::Deref, rc::Rc};

use rpds::Vector;

use crate::{Result, eval::Environment, meta::MetaContext, term::TermPtr, value::ValuePtr};

#[derive(Debug, Copy, Clone)]
pub struct WithSpan<T> {
    data: T,
    pub span: (usize, usize),
}

impl<T: PartialEq> PartialEq for WithSpan<T> {
    fn eq(&self, other: &Self) -> bool {
        self.data == other.data
    }
}

impl<T: Eq> Eq for WithSpan<T> {}

impl<T> std::hash::Hash for WithSpan<T>
where
    T: std::hash::Hash,
{
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.data.hash(state);
    }
}

impl<T> Deref for WithSpan<T> {
    type Target = T;

    fn deref(&self) -> &Self::Target {
        &self.data
    }
}

impl<T> WithSpan<T> {
    pub fn data(&self) -> &T {
        &self.data
    }
}

#[derive(Clone, Eq)]
#[repr(transparent)]
pub struct UniqueName(Rc<WithSpan<ustr::Ustr>>);

impl Display for UniqueName {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.name().fmt(f)
    }
}

impl UniqueName {
    pub fn new<T: Into<ustr::Ustr>>(name: T, span: (usize, usize)) -> Self {
        Self(Rc::new(WithSpan {
            data: name.into(),
            span,
        }))
    }
    pub fn fresh(span: (usize, usize)) -> Self {
        Self(Rc::new(WithSpan {
            data: "$x".into(),
            span,
        }))
    }
    pub fn refresh(&self) -> Self {
        Self(Rc::new(*self.0))
    }
    pub fn fresh_in<F>(&self, lookup: F) -> Self
    where
        F: FnOnce(&Self) -> bool,
    {
        if lookup(self) {
            Self::fresh(self.0.span)
        } else {
            self.clone()
        }
    }
    pub fn span(&self) -> (usize, usize) {
        self.0.span
    }
    fn name(&self) -> ustr::Ustr {
        self.0.data
    }
}

impl std::fmt::Debug for UniqueName {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}@{:?}", **self.0, Rc::as_ptr(&self.0))
    }
}

impl PartialEq for UniqueName {
    fn eq(&self, other: &Self) -> bool {
        Rc::ptr_eq(&self.0, &other.0)
    }
}

impl std::hash::Hash for UniqueName {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        Rc::as_ptr(&self.0).hash(state);
    }
}

#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub enum Icit {
    Expl,
    Impl,
}

impl std::fmt::Display for Icit {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Icit::Expl => write!(f, "explicit"),
            Icit::Impl => write!(f, "implicit"),
        }
    }
}

pub type Pruning = Vector<(UniqueName, Icit)>;
pub type Spine = Vector<(ValuePtr, Icit)>;

#[derive(Debug, Clone)]
pub struct Closure {
    env: RefCell<Environment>,
    body: TermPtr,
}

pub fn empty_spine() -> Spine {
    thread_local! {
        static EMPTY_SPINE: Spine = Vector::new();
    }
    EMPTY_SPINE.with(|spine| spine.clone())
}

pub fn with_span<T>(data: T, span: (usize, usize)) -> Rc<WithSpan<T>> {
    Rc::new(WithSpan { data, span })
}

pub fn with_span_as<T, X, Y>(data: T, target: X) -> Rc<WithSpan<T>>
where
    X: AsRef<WithSpan<Y>>,
{
    Rc::new(WithSpan {
        data,
        span: target.as_ref().span,
    })
}

impl Closure {
    pub fn new(env: Environment, body: TermPtr) -> Self {
        let env = RefCell::new(env);
        Self { env, body }
    }
    pub fn apply(&self, name: UniqueName, arg: ValuePtr, meta: &MetaContext) -> Result<ValuePtr> {
        let mut env = self.env.borrow_mut();
        env.with_var(name, arg, |env| env.evaluate(self.body.clone(), meta))
    }
}
