use std::{cell::RefCell, ops::Deref, rc::Rc};

use rpds::Vector;
use ustr::Ustr;

use crate::{eval::Environment, meta::MetaContext, term::TermPtr, value::ValuePtr, Result};

#[derive(Debug, Copy, Clone)]
pub struct WithSpan<T> {
    data: T,
    pub span: Span,
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

#[repr(transparent)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub struct DBIdx(pub(crate) usize);

impl DBIdx {
    pub fn zero() -> Self {
        Self(0)
    }
    pub fn next(self) -> Self {
        Self(self.0 + 1)
    }
    pub fn to_level(self, env_len: usize) -> DBLvl {
        DBLvl(env_len - self.0 - 1)
    }
}

#[repr(transparent)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub struct DBLvl(pub(crate) usize);

impl DBLvl {
    pub fn zero() -> Self {
        Self(0)
    }
    pub fn next(self) -> Self {
        Self(self.0 + 1)
    }
    pub fn to_index(self, level: Self) -> DBIdx {
        DBIdx(level.0 - self.0 - 1)
    }
}

#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
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

pub type Pruning = Vector<Option<Icit>>;
pub type Spine = Vector<(ValuePtr, Icit)>;
pub type Name = WithSpan<Ustr>;
pub type Span = std::range::Range<usize>;

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

pub fn with_span<T>(data: T, span: Span) -> Rc<WithSpan<T>> {
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
    pub fn apply(&self, arg: ValuePtr, meta: &MetaContext) -> Result<ValuePtr> {
        let mut env = self.env.borrow_mut();
        env.with_var(arg, |env| env.evaluate(self.body.clone(), meta))
    }
}
