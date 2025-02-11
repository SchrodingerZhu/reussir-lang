#![allow(unused)]
use std::{cell::RefCell, ops::Deref, path::Path};

use bumpalo::Bump;
use chumsky::{
    Parser, container::Container, error::Rich, extra::Full, input::Input, span::SimpleSpan,
};
use rustc_hash::FxHashMapRand;
use smallvec::SmallVec;
use thiserror::Error;
use r#type::Record;

mod expr;
mod func;
mod lexer;
mod r#type;

struct SmallCollector<T, const N: usize>(SmallVec<T, N>);

impl<T, const N: usize> Default for SmallCollector<T, N> {
    fn default() -> Self {
        Self(Default::default())
    }
}

impl<T: Default, const N: usize> Container<T> for SmallCollector<T, N> {
    fn with_capacity(n: usize) -> Self {
        Self(SmallVec::with_capacity(n))
    }

    fn push(&mut self, item: T) {
        self.0.push(item);
    }
}

#[derive(Debug, Copy, Clone)]
pub struct WithSpan<T>(pub T, pub SimpleSpan);

impl<T: PartialEq> PartialEq for WithSpan<T> {
    fn eq(&self, other: &Self) -> bool {
        self.0 == other.0
    }
}

impl<T: Eq> Eq for WithSpan<T> {}

impl<T> std::hash::Hash for WithSpan<T>
where
    T: std::hash::Hash,
{
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.0.hash(state);
    }
}

impl<T> Deref for WithSpan<T> {
    type Target = T;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

type RecordStore<'ctx> = FxHashMapRand<QualifiedName<'ctx>, WithSpan<&'ctx Record<'ctx>>>;

pub struct Context<'ctx> {
    arena: Bump,
    input: Option<std::path::PathBuf>,
    src: String,
    records: RefCell<RecordStore<'ctx>>,
    scope: Vec<&'ctx str>,
}

#[derive(Debug, Copy, Clone, Eq, PartialEq, Hash)]
pub struct QualifiedName<'ctx>(&'ctx [&'ctx str], &'ctx str);

impl QualifiedName<'_> {
    pub fn qualifier(&self) -> &[&str] {
        self.0
    }
    pub fn is_unqualified(&self) -> bool {
        self.0.is_empty()
    }
    pub fn basename(&self) -> &str {
        self.1
    }
}

#[derive(Error, Debug)]
pub enum Error {
    #[error("failed to read source from input {0}")]
    IoError(#[from] std::io::Error),
}

type Result<T> = std::result::Result<T, Error>;

impl<'ctx> Context<'ctx> {
    pub fn from_file<P: AsRef<Path>>(input: P) -> Result<Self> {
        let input = input.as_ref().to_owned();
        let src = std::fs::read_to_string(&input)?;
        let input = Some(input);
        Ok(Self::new(input, src))
    }

    pub fn from_src<S: AsRef<str>>(src: S) -> Self {
        Self::new(None, src.as_ref().to_string())
    }

    pub fn alloc_str<S: AsRef<str>>(&self, src: S) -> &str {
        self.arena.alloc_str(src.as_ref())
    }

    fn new(input: Option<std::path::PathBuf>, src: String) -> Self {
        Self {
            arena: Bump::new(),
            input,
            src,
            records: RefCell::new(FxHashMapRand::default()),
            scope: Vec::new(),
        }
    }
    pub fn new_qualified_name<I>(
        &'ctx self,
        qualifiers: I,
        basename: &'ctx str,
    ) -> QualifiedName<'ctx>
    where
        I: IntoIterator<Item = &'ctx str>,
        I::IntoIter: ExactSizeIterator,
    {
        let qualifiers = self.arena.alloc_slice_fill_iter(qualifiers);
        QualifiedName(qualifiers, basename)
    }
}
