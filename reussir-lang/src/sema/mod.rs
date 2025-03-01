#![allow(unused)]
mod term;
use std::{cell::RefCell, rc::Rc};

use chumsky::span::SimpleSpan;
use norm::ValuePtr;
use rustc_hash::FxHashMapRand;
use ustr::Ustr;

use crate::syntax::{self, WithSpan};
use term::TermPtr;
mod norm;

#[derive(Debug, Clone, Copy, Eq, PartialEq, Hash)]
pub enum FieldName {
    Idx(usize),
    Name(Ustr),
}

#[derive(Debug, Clone, Eq, PartialEq, Hash)]
pub struct QualifiedName(Box<[Ustr]>, Ustr);

impl QualifiedName {
    pub fn new(name: syntax::QualifiedName) -> Self {
        let qualifier = name.qualifier().iter().copied().map(ustr::ustr).collect();
        let basename = ustr::Ustr::from(name.basename());
        Self(qualifier, basename)
    }

    pub fn qualifier(&self) -> &[Ustr] {
        &self.0
    }

    pub fn basename(&self) -> &Ustr {
        &self.1
    }
}

#[derive(Clone)]
pub struct Context {
    functions: FxHashMapRand<QualifiedName, TermPtr>,
    meta_variable: RefCell<Vec<MetaEntry>>,
}

#[derive(Clone)]
pub enum MetaEntry {
    Unsolved(SimpleSpan),
    Solved(ValuePtr),
}

impl Context {
    pub fn new() -> Self {
        Self {
            functions: Default::default(),
            meta_variable: RefCell::new(Vec::new()),
        }
    }
    pub fn fresh_meta(&self, span: SimpleSpan) -> usize {
        let mut v = self.meta_variable.borrow_mut();
        let res = v.len();
        v.push(MetaEntry::Unsolved(span));
        res
    }
    pub fn lookup_meta(&self, idx: usize) -> Option<MetaEntry> {
        self.meta_variable.borrow().get(idx).cloned()
    }
}

#[derive(Clone, Eq)]
#[repr(transparent)]
pub struct UniqueName(Rc<WithSpan<ustr::Ustr>>);

impl UniqueName {
    fn new<T: Into<ustr::Ustr>>(name: T, span: SimpleSpan) -> Self {
        Self(Rc::new(WithSpan(name.into(), span)))
    }
    fn fresh(span: SimpleSpan) -> Self {
        Self(Rc::new(WithSpan("$x".into(), span)))
    }
    fn fresh_in<F>(&self, lookup: F) -> Self
    where
        F: FnOnce(&Self) -> bool,
    {
        if lookup(self) {
            Self::fresh(self.span())
        } else {
            self.clone()
        }
    }
    fn span(&self) -> SimpleSpan {
        self.0.1
    }
    fn name(&self) -> ustr::Ustr {
        self.0.0
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

#[cfg(test)]
mod test {
    use super::*;
    #[test]
    fn it_creates_qualified_name() {
        let fake_name = syntax::QualifiedName::new(&["std", "test"], "test");
        _ = QualifiedName::new(fake_name);
    }
}
