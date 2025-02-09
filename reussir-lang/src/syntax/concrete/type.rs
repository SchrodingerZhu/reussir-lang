use std::collections::HashMap;

use crate::syntax::concrete::r#type::Error::ExpectedFnType;
use crate::syntax::concrete::{Decl, Def, Expr, File};
use crate::syntax::r#abstract::convert::convert;
use crate::syntax::r#abstract::{
    Decl as WellTypedDecl, Def as WellTypedDef, File as WellTypedFile, Inferred, Term,
};
use crate::syntax::{Ctor, CtorParams, DataDef, FnDef, FnSig, Ident, Param, ID};

#[allow(dead_code)]
#[derive(Debug)]
enum Error<'src> {
    MismatchedType { want: Term<'src>, got: Term<'src> },
    MismatchedEffect { want: Term<'src>, got: Term<'src> },
    ExpectedFnType { typ: Term<'src> },
}

#[allow(dead_code)]
#[derive(Default)]
struct Checker<'src> {
    globals: HashMap<ID, WellTypedDef<'src>>,
    locals: HashMap<ID, Term<'src>>,
}

#[allow(dead_code)]
impl<'src> Checker<'src> {
    pub fn file(mut self, f: File<'src>) -> Result<WellTypedFile<'src>, Error<'src>> {
        Ok(WellTypedFile {
            decls: f
                .decls
                .into_vec()
                .into_iter()
                .map(|d| self.decl(d))
                .collect::<Result<Vec<_>, _>>()?
                .into_iter()
                .map(|name| WellTypedDecl {
                    name,
                    def: self.globals.remove(&name.id).unwrap(),
                })
                .collect(),
        })
    }

    fn decl(&mut self, decl: Decl<'src>) -> Result<Ident<'src>, Error<'src>> {
        self.locals.clear();
        let id = decl.name.id;
        match decl.def {
            Def::Fn(d) => self.fn_def(id, d)?,
            Def::Data(d) => self.data_def(id, d)?,
        }
        Ok(decl.name)
    }

    fn fn_def(&mut self, id: ID, def: FnDef<'src, Expr<'src>>) -> Result<(), Error<'src>> {
        let FnDef {
            sig:
                FnSig {
                    typ_params,
                    val_params,
                    eff,
                    ret,
                },
            body,
        } = def;
        let typ_params = self.params(&typ_params)?;
        let val_params = self.params(&val_params)?;
        let eff = self.check_type(&eff)?;
        let ret = self.check_type(&ret)?;
        let expected_eff = *eff.clone();
        let expected_typ = *ret.clone();
        self.globals.insert(
            id,
            WellTypedDef::UndefFn(FnSig {
                typ_params,
                val_params,
                eff,
                ret,
            }),
        );

        let body = self.check(&body, expected_eff, expected_typ)?;
        let sig = match self.globals.remove(&id).unwrap() {
            WellTypedDef::UndefFn(sig) => sig,
            _ => unreachable!(),
        };
        self.globals
            .insert(id, WellTypedDef::Fn(FnDef { sig, body }));

        Ok(())
    }

    fn data_def(&mut self, id: ID, def: DataDef<'src, Expr<'src>>) -> Result<(), Error<'src>> {
        let DataDef { typ_params, ctors } = def;
        let typ_params = self.params(&typ_params)?;
        let ctors = ctors
            .into_vec()
            .into_iter()
            .map(|Ctor { name, params }| {
                let params = match params {
                    CtorParams::None => CtorParams::None,
                    CtorParams::Unnamed(ts) => CtorParams::Unnamed(
                        ts.iter()
                            .map(|t| self.check_type(t))
                            .collect::<Result<_, _>>()?,
                    ),
                    CtorParams::Named(ps) => CtorParams::Named(self.params(&ps)?),
                };
                Ok(Ctor { name, params })
            })
            .collect::<Result<_, _>>()?;
        self.globals
            .insert(id, WellTypedDef::Data(DataDef { typ_params, ctors }));
        Ok(())
    }

    fn params(
        &mut self,
        ps: &[Param<'src, Expr<'src>>],
    ) -> Result<Box<[Param<'src, Term<'src>>]>, Error<'src>> {
        ps.iter().map(|p| self.param(p)).collect()
    }

    fn param(
        &mut self,
        Param { name, typ }: &Param<'src, Expr<'src>>,
    ) -> Result<Param<'src, Term<'src>>, Error<'src>> {
        let name = *name;
        let typ = self.check_type(typ)?;
        self.locals.insert(name.id, *typ.clone());
        Ok(Param { name, typ })
    }

    fn check_type(&mut self, typ: &Expr<'src>) -> Result<Box<Term<'src>>, Error<'src>> {
        self.check(typ, Term::Pure, Term::Type)
    }

    fn guarded_check(
        &mut self,
        ctx: &[Param<'src, Term<'src>>],
        e: &Expr<'src>,
        eff: Term<'src>,
        typ: Term<'src>,
    ) -> Result<Box<Term<'src>>, Error<'src>> {
        ctx.iter().for_each(|p| {
            self.locals.insert(p.name.id, *p.typ.clone());
        });

        let term = self.check(e, eff, typ)?;

        ctx.iter().for_each(|p| {
            self.locals.remove(&p.name.id);
        });

        Ok(term)
    }

    fn check(
        &mut self,
        e: &Expr<'src>,
        eff: Term<'src>,
        typ: Term<'src>,
    ) -> Result<Box<Term<'src>>, Error<'src>> {
        if let Expr::Fn { params, body } = e {
            return match typ {
                Term::FnType {
                    param_types,
                    eff,
                    ret,
                } => {
                    let params = params.clone();
                    let ctx = params
                        .iter()
                        .map(Clone::clone)
                        .zip(param_types.into_vec())
                        .map(|(name, typ)| Param { name, typ })
                        .collect::<Box<_>>();
                    let body = self.guarded_check(&ctx, body, *eff, *ret)?;
                    Ok(Box::new(Term::Fn { params, body }))
                }
                typ => Err(Error::ExpectedFnType { typ }),
            };
        }

        let Inferred {
            term,
            eff: got_eff,
            typ: got_typ,
        } = self.infer(e)?;
        convert(&eff, &got_eff)
            .then_some(())
            .ok_or(Error::MismatchedEffect {
                want: eff,
                got: got_eff,
            })?;
        convert(&typ, &got_typ)
            .then_some(())
            .ok_or(Error::MismatchedType {
                want: typ,
                got: got_typ,
            })?;

        Ok(Box::new(term))
    }

    fn infer(&mut self, e: &Expr<'src>) -> Result<Inferred<'src>, Error<'src>> {
        use Expr::*;
        Ok(match e {
            Ident(i) => self
                .locals
                .get(&i.id)
                .cloned()
                .map(|ty| Inferred::pure(Term::Ident(*i), ty))
                .unwrap_or_else(|| self.globals.get(&i.id).unwrap().to_inferred(*i)),
            Type => Inferred::r#type(Term::Type),
            NoneType => Inferred::r#type(Term::NoneType),
            None => Inferred::pure(Term::None, Term::NoneType),
            Boolean => Inferred::r#type(Term::Boolean),
            False => Inferred::pure(Term::False, Term::Boolean),
            True => Inferred::pure(Term::True, Term::Boolean),
            String => Inferred::r#type(Term::String),
            Str(s) => Inferred::pure(Term::Str(s), Term::String),
            F32 => Inferred::r#type(Term::F32),
            F64 => Inferred::r#type(Term::F64),
            Float(v) => Inferred::pure(Term::Float(*v), Term::F64),
            FnType {
                param_types,
                eff,
                ret,
            } => Inferred::r#type(Term::FnType {
                param_types: param_types
                    .iter()
                    .map(|t| self.check_type(t))
                    .collect::<Result<_, _>>()?,
                eff: self.check_type(eff)?,
                ret: self.check_type(ret)?,
            }),
            Fn { .. } => unreachable!(),
            Call { f, args } => {
                let Inferred { term, eff, typ } = self.infer(f)?;
                let (arg_terms, (arg_effs, arg_types)): (Vec<_>, (Vec<_>, Vec<_>)) = args
                    .iter()
                    .map(|a| self.infer(a))
                    .collect::<Result<Vec<_>, _>>()?
                    .into_iter()
                    .map(|Inferred { term, eff, typ }| (Box::new(term), (eff, typ)))
                    .unzip();
                match typ {
                    Term::FnType {
                        param_types, ret, ..
                    } => {
                        param_types.into_vec().into_iter().zip(arg_types).try_fold(
                            (),
                            |_, (want, got)| {
                                let want = *want;
                                convert(&want, &got)
                                    .then_some(())
                                    .ok_or(Error::MismatchedType { want, got })
                            },
                        )?;
                        arg_effs.into_iter().try_fold((), |_, got| {
                            let want = eff.clone();
                            convert(&want, &got)
                                .then_some(())
                                .ok_or(Error::MismatchedEffect { want, got })
                        })?;
                        Inferred {
                            term: Term::Call {
                                f: Box::new(term),
                                args: arg_terms.into_boxed_slice(),
                            },
                            eff,
                            typ: *ret,
                        }
                    }
                    typ => return Err(ExpectedFnType { typ }),
                }
            }
            Pure => Inferred::r#type(Term::Pure),
        })
    }
}
