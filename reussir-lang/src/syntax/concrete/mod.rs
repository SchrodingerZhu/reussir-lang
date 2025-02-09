use crate::syntax::{DataDef, FnDef, Ident, Syntax};

mod scope;
mod r#type;

#[allow(dead_code)]
#[derive(Debug)]
pub struct File<'src> {
    pub decls: Box<[Decl<'src>]>,
}

#[allow(dead_code)]
#[derive(Debug)]
pub struct Decl<'src> {
    pub name: Ident<'src>,
    pub def: Def<'src>,
}

#[allow(dead_code)]
#[derive(Debug)]
pub enum Def<'src> {
    Fn(FnDef<'src, Expr<'src>>),
    Data(DataDef<'src, Expr<'src>>),
}

#[allow(dead_code)]
#[derive(Debug, Clone)]
pub enum Expr<'src> {
    Ident(Ident<'src>),

    Type,

    NoneType,
    None,

    Boolean,
    False,
    True,

    String,
    Str(&'src str),

    F32,
    F64,
    Float(f64),

    FnType {
        param_types: Box<[Box<Self>]>,
        eff: Box<Self>,
        ret: Box<Self>,
    },
    Fn {
        params: Box<[Ident<'src>]>,
        body: Box<Self>,
    },
    Call {
        f: Box<Self>,
        args: Box<[Box<Self>]>,
    },

    Pure,
}

impl Syntax for Expr<'_> {}
