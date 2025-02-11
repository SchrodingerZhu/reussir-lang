use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};

pub mod r#abstract;
pub mod concrete;
pub mod surface;

pub type ID = u64;

pub type NameMap<'src> = HashMap<&'src str, ID>;

#[derive(Debug, Copy, Clone)]
pub struct Ident<'src> {
    pub raw: &'src str,
    pub id: ID,
}

pub fn fresh() -> ID {
    static NEXT_ID: AtomicU64 = AtomicU64::new(1);
    NEXT_ID.fetch_add(1, Ordering::Relaxed)
}

pub trait Syntax {}

#[allow(dead_code)]
#[derive(Debug, Clone)]
pub struct Param<'src, T: Syntax> {
    pub name: Ident<'src>,
    pub typ: Box<T>,
}

#[allow(dead_code)]
#[derive(Debug)]
pub struct FnSig<'src, T: Syntax> {
    pub typ_params: Box<[Param<'src, T>]>,
    pub val_params: Box<[Param<'src, T>]>,
    pub eff: Box<T>,
    pub ret: Box<T>,
}

#[allow(dead_code)]
#[derive(Debug)]
pub struct FnDef<'src, T: Syntax> {
    pub sig: FnSig<'src, T>,
    pub body: Box<T>,
}

#[allow(dead_code)]
#[derive(Debug)]
pub struct DataDef<'src, T: Syntax> {
    pub typ_params: Box<[Param<'src, T>]>,
    pub ctors: Box<[Ctor<'src, T>]>,
}

#[allow(dead_code)]
#[derive(Debug)]
pub struct Ctor<'src, T: Syntax> {
    pub name: Ident<'src>,
    pub params: CtorParams<'src, T>,
}

#[allow(dead_code)]
#[derive(Debug)]
pub enum CtorParams<'src, T: Syntax> {
    None,
    Unnamed(Box<[Box<T>]>),
    Named(Box<[Param<'src, T>]>),
}

impl<T: Syntax> Default for CtorParams<'_, T> {
    fn default() -> Self {
        Self::None
    }
}
