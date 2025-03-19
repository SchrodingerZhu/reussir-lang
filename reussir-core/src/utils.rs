use std::{fmt::Display, hash::Hash, ops::Deref, rc::Rc};

use rpds::{HashTrieMap, Vector};

use crate::{
    meta::MetaContext,
    value::{self, ValuePtr},
};

#[derive(Debug, Copy, Clone)]
pub struct WithSpan<T> {
    data: T,
    pub start: usize,
    pub end: usize,
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

#[derive(Clone, Eq)]
#[repr(transparent)]
pub struct UniqueName(Rc<WithSpan<ustr::Ustr>>);

impl Display for UniqueName {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.name().fmt(f)
    }
}

impl UniqueName {
    pub fn new<T: Into<ustr::Ustr>>(name: T, start: usize, end: usize) -> Self {
        Self(Rc::new(WithSpan {
            data: name.into(),
            start,
            end,
        }))
    }
    pub fn fresh(start: usize, end: usize) -> Self {
        Self(Rc::new(WithSpan {
            data: "$x".into(),
            start,
            end,
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
            Self::fresh(self.0.start, self.0.end)
        } else {
            self.clone()
        }
    }
    pub fn start(&self) -> usize {
        self.0.start
    }
    pub fn end(&self) -> usize {
        self.0.end
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

pub type Pruning = Vector<(UniqueName, Icit)>;
pub type Environment = HashTrieMap<UniqueName, ValuePtr>;
pub type Spine = Vector<(ValuePtr, Icit)>;

#[derive(Debug, Clone)]
pub struct Closure {
    pub env: Environment,
    pub body: ValuePtr,
}

pub fn empty_spine() -> Spine {
    thread_local! {
        static EMPTY_SPINE: Spine = Vector::new();
    }
    EMPTY_SPINE.with(|spine| spine.clone())
}

pub fn with_span<T>(data: T, start: usize, end: usize) -> Rc<WithSpan<T>> {
    Rc::new(WithSpan { data, start, end })
}

pub fn with_var<F, R>(env: &mut Environment, name: UniqueName, value: ValuePtr, f: F) -> R
where
    F: FnOnce(&mut Environment) -> R,
{
    env.insert_mut(name.clone(), value.clone());
    let res = f(env);
    env.remove_mut(&name);
    res
}
