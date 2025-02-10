#![allow(unused)]
use std::path::Path;

use bumpalo::Bump;
use rustc_hash::FxHashMapRand;
use thiserror::Error;
use r#type::Record;

mod expr;
mod func;
mod r#type;

struct Context<'ctx> {
    arena: Bump,
    input: Option<std::path::PathBuf>,
    src: String,
    records: FxHashMapRand<QualifiedName<'ctx>, Record<'ctx>>,
    scope: Vec<&'ctx str>,
}

#[derive(Debug, Copy, Clone, Eq, PartialEq, Hash)]
struct QualifiedName<'ctx>(&'ctx [&'ctx str], &'ctx str);

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

    fn new(input: Option<std::path::PathBuf>, src: String) -> Self {
        Self {
            arena: Bump::new(),
            input,
            src,
            records: FxHashMapRand::default(),
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
