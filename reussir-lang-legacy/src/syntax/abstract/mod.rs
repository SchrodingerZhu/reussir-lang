use crate::syntax::r#abstract::rename::rename;
use crate::syntax::{DataDef, FnDef, FnSig, Ident, Param, Syntax};

pub mod convert;
pub mod inline;
mod rename;

#[allow(dead_code)]
pub struct Inferred<'src> {
    pub term: Term<'src>,
    pub eff: Term<'src>,
    pub typ: Term<'src>,
}

impl<'src> Inferred<'src> {
    pub fn pure(term: Term<'src>, typ: Term<'src>) -> Self {
        let eff = Term::Pure;
        Self { term, eff, typ }
    }

    pub fn r#type(typ: Term<'src>) -> Self {
        Self::pure(typ, Term::Type)
    }
}

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
    Fn(FnDef<'src, Term<'src>>),
    UndefFn(FnSig<'src, Term<'src>>),
    Data(DataDef<'src, Term<'src>>),
}

impl<'src> Def<'src> {
    pub fn to_inferred(&self, ident: Ident<'src>) -> Inferred<'src> {
        let (mut term, mut eff, mut typ) = match self {
            Def::Fn(d) => (d.into(), *d.sig.eff.clone(), Term::from(&d.sig)),
            Def::UndefFn(sig) => (sig.to_term(ident), *sig.eff.clone(), Term::from(sig)),
            Def::Data(_) => todo!(),
        };
        rename(&mut term);
        rename(&mut eff);
        rename(&mut typ);
        Inferred { term, eff, typ }
    }
}

impl<'src> From<&FnDef<'src, Term<'src>>> for Term<'src> {
    fn from(d: &FnDef<'src, Term<'src>>) -> Self {
        d.sig.typ_params.iter().rfold(
            Term::Fn {
                params: d.sig.val_params.iter().map(|p| p.name).collect(),
                body: d.body.clone(),
            },
            |body, p| Term::GenericFn {
                param: p.clone(),
                body: Box::new(body),
            },
        )
    }
}

impl<'src> FnSig<'src, Term<'src>> {
    fn to_term(&self, ident: Ident<'src>) -> Term<'src> {
        self.typ_params.iter().rfold(
            Term::Fn {
                params: self.val_params.iter().map(|p| p.name).collect(),
                body: Box::new(Term::Ident(ident)),
            },
            |body, p| Term::GenericFn {
                param: p.clone(),
                body: Box::new(body),
            },
        )
    }
}

impl<'src> From<&FnSig<'src, Term<'src>>> for Term<'src> {
    fn from(sig: &FnSig<'src, Term<'src>>) -> Self {
        let FnSig {
            typ_params,
            val_params,
            eff,
            ret,
        } = sig;
        typ_params.iter().rfold(
            Term::FnType {
                param_types: val_params.iter().map(|p| p.typ.clone()).collect(),
                eff: eff.clone(),
                ret: ret.clone(),
            },
            |body, p| Term::GenericFnType {
                param: p.clone(),
                body: Box::new(body),
            },
        )
    }
}

#[allow(dead_code)]
#[derive(Debug, Clone)]
pub enum Term<'src> {
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

    GenericFnType {
        param: Param<'src, Term<'src>>,
        body: Box<Self>,
    },
    GenericFn {
        param: Param<'src, Term<'src>>,
        body: Box<Self>,
    },
}

impl Syntax for Term<'_> {}
