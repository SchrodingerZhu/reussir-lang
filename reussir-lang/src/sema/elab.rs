//!
//! Bidirectional Elaboration Type Checking

use std::rc::Rc;

use chumsky::span::SimpleSpan;
use rpds::Queue;
use rustc_hash::FxRandomState;
use thiserror::Error;

use crate::syntax::WithSpan;

use super::{
    Context, UniqueName,
    eval::{Environment, evaluate, force, quote},
    term::{Term, TermPtr},
    unify::unify,
    value::{Closure, Map, Value, ValuePtr, empty_spine},
};

#[derive(Debug, Clone, Error)]
pub enum Error {
    #[error("Failed to unify {} with {} due to {reason}", ***lhs, ***rhs)]
    UnificationFailure {
        lhs: TermPtr,
        rhs: TermPtr,
        reason: super::unify::Error,
    },
    #[error("Variable {} is not defined", .0.name())]
    UndefinedVariable(UniqueName),
    #[error("Cannot infer type for lambda with named argument")]
    InferNamedLambda(SimpleSpan),
    #[error(
        "Mismatched implicity, expected {}, found {}",
        implicity_to_string(*expected),
        implicity_to_string(*actual)
    )]
    ImplicityMismatch {
        span: SimpleSpan,
        expected: bool,
        actual: bool,
    },
}

fn implicity_to_string(implicit: bool) -> &'static str {
    if implicit { "implicit" } else { "explicit" }
}

struct ElabContext<'a> {
    env: Environment<'a>,
    bounded: Queue<UniqueName>,
    type_info: Map<UniqueName, ValuePtr>,
}

fn var(name: UniqueName) -> ValuePtr {
    Rc::new(WithSpan(
        Value::Rigid(name.clone(), empty_spine()),
        name.span(),
    ))
}

impl<'a> ElabContext<'a> {
    fn new(global: &'a Context) -> Self {
        let env = Environment::new(global);
        Self {
            env,
            bounded: Queue::new(),
            type_info: Map::new_with_hasher_and_ptr_kind(FxRandomState::new()),
        }
    }
    fn fresh_meta(&self, span: SimpleSpan) -> TermPtr {
        let meta = self.env.global().fresh_meta(span);
        Rc::new(WithSpan(
            Term::InsertedMeta(meta, self.bounded.clone()),
            span,
        ))
    }
    fn insert_lambda(&self, (term, ty): (TermPtr, ValuePtr)) -> (TermPtr, ValuePtr) {
        if matches!(**term, Term::Lambda { implicit: true, .. }) {
            (term, ty)
        } else {
            self.insert_app((term, ty))
        }
    }

    fn insert_app(&self, (mut term, mut ty): (TermPtr, ValuePtr)) -> (TermPtr, ValuePtr) {
        while let Value::Pi {
            name,
            arg,
            body,
            implicit: true,
        } = {
            ty = force(ty, self.env.global());
            &**ty
        } {
            let meta_term = self.fresh_meta(name.span());
            let meta_value = evaluate(self.env.clone(), meta_term.clone());
            let term_span = term.1;
            term = Rc::new(WithSpan(Term::App(term, meta_term, true), term_span));
            ty = body(name.clone(), meta_value, self.env.global());
        }
        (term, ty)
    }
    fn bind(&self, name: UniqueName, ty: ValuePtr) -> Self {
        let value = Rc::new(WithSpan(
            Value::Rigid(name.clone(), empty_spine()),
            name.span(),
        ));
        let env = self.env.define(name.clone(), value);
        let bounded = self.bounded.enqueue(name.clone());
        let type_info = self.type_info.insert(name, ty);
        Self {
            env,
            bounded,
            type_info,
        }
    }
    fn define(&self, name: UniqueName, value: ValuePtr, ty: ValuePtr) -> Self {
        let env = self.env.define(name.clone(), value);
        let bounded = self.bounded.clone();
        let type_info = self.type_info.insert(name, ty);
        Self {
            env,
            bounded,
            type_info,
        }
    }
    fn evaluate(&self, term: TermPtr) -> ValuePtr {
        evaluate(self.env.clone(), term)
    }
    fn try_unify(&self, lhs: ValuePtr, rhs: ValuePtr) -> Result<(), Error> {
        let global = self.env.global();
        unify(lhs.clone(), rhs.clone(), global).map_err(|reason| Error::UnificationFailure {
            lhs: quote(lhs, global),
            rhs: quote(rhs, global),
            reason,
        })
    }

    fn check(&self, term: TermPtr, ty: ValuePtr) -> Result<TermPtr, Error> {
        tracing::trace!(
            "checking {} against {}",
            **term,
            **quote(ty.clone(), self.env.global())
        );
        match &**term {
            Term::Let {
                name,
                ty: var_ty,
                binding,
                body,
            } => {
                let var_ty = var_ty
                    .as_ref()
                    .map(|ty| self.check_universe(ty.clone()))
                    .unwrap_or_else(|| Ok(self.fresh_meta(ty.1)))?;
                let var_ty_value = self.evaluate(var_ty.clone());
                let var = self.check(binding.clone(), ty.clone())?;
                let var_value = self.evaluate(var.clone());
                let body = self
                    .define(name.clone(), var_value, var_ty_value)
                    .check(body.clone(), ty)?;
                return Ok(Rc::new(WithSpan(
                    Term::Let {
                        name: name.clone(),
                        ty: Some(var_ty),
                        binding: var,
                        body,
                    },
                    term.1,
                )));
            }
            Term::Hole => return Ok(self.fresh_meta(term.1)),
            Term::Lambda {
                binding,
                body: body_lam,
                implicit: implicit_lam,
            } => {
                if let Value::Pi {
                    name,
                    arg,
                    body: body_pi,
                    implicit: implicit_pi,
                } = &**ty
                {
                    if (*implicit_pi && name == binding) || implicit_pi == implicit_lam {
                        let span = term.1;
                        let body = self.bind(binding.clone(), arg.clone()).check(
                            body_lam.clone(),
                            body_pi(name.clone(), var(binding.clone()), self.env.global()),
                        )?;
                        return Ok(Rc::new(WithSpan(
                            Term::Lambda {
                                binding: binding.clone(),
                                body,
                                implicit: *implicit_pi,
                            },
                            span,
                        )));
                    }
                }
            }
            _ => (),
        }
        if let Value::Pi {
            name,
            arg,
            body,
            implicit: true,
        } = &**ty
        {
            let span = term.1;
            let body = self.bind(name.clone(), arg.clone()).check(
                term,
                body(name.clone(), var(name.clone()), self.env.global()),
            )?;
            return Ok(Rc::new(WithSpan(
                Term::Lambda {
                    binding: name.clone(),
                    body,
                    implicit: true,
                },
                span,
            )));
        }

        let (term, inferred) = self.insert_lambda(self.infer(term)?);
        self.try_unify(ty, inferred)?;
        Ok(term)
    }

    fn quote_closure(&self, value: ValuePtr) -> Rc<Closure> {
        let body = quote(value, self.env.global());
        self.closure(body)
    }

    fn closure(&self, body: TermPtr) -> Rc<Closure> {
        let env = self.env.bindings();
        Rc::new(move |name, val, glb| {
            let env = Environment::new_with(glb, env.clone()).define(name, val);
            evaluate(env, body.clone())
        })
    }

    fn check_universe(&self, term: TermPtr) -> Result<TermPtr, Error> {
        let span = term.1;
        self.check(term, Rc::new(WithSpan(Value::Universe, span)))
    }

    fn infer(&self, term: TermPtr) -> Result<(TermPtr, ValuePtr), Error> {
        tracing::trace!("inferring {}", **term);
        match &**term {
            Term::Var(x) => {
                if let Some(ty) = self.type_info.get(x) {
                    Ok((term, ty.clone()))
                } else {
                    tracing::error!(
                        "{:?} not found in {:?}",
                        x,
                        self.type_info.keys().collect::<Vec<_>>()
                    );
                    Err(Error::UndefinedVariable(x.clone()))
                }
            }
            Term::Lambda {
                binding,
                body,
                implicit,
            } => {
                let meta = self.evaluate(self.fresh_meta(binding.span()));
                let ctx = self.bind(binding.clone(), meta.clone());
                let (body, ty) = ctx.insert_lambda(ctx.infer(body.clone())?);
                let term = Rc::new(WithSpan(
                    Term::Lambda {
                        binding: binding.clone(),
                        body,
                        implicit: *implicit,
                    },
                    term.1,
                ));
                let span = ty.1;
                let ty = Rc::new(WithSpan(
                    Value::Pi {
                        name: binding.clone(),
                        arg: meta,
                        body: self.quote_closure(ty),
                        implicit: *implicit,
                    },
                    span,
                ));
                Ok((term, ty))
            }
            Term::App(lhs, rhs, implicity) => {
                let (mut lhs, mut lhs_ty) = self.infer(lhs.clone())?;
                if !*implicity {
                    (lhs, lhs_ty) = self.insert_app((lhs, lhs_ty));
                }
                lhs_ty = force(lhs_ty, self.env.global());
                let (arg_ty, name, body) = if let Value::Pi {
                    name,
                    arg,
                    body,
                    implicit: ipi,
                } = &**lhs_ty
                {
                    if *ipi != *implicity {
                        return Err(Error::ImplicityMismatch {
                            span: term.1,
                            expected: *ipi,
                            actual: *implicity,
                        });
                    }
                    (arg.clone(), name.clone(), body.clone())
                } else {
                    // construct a meta pi and unify
                    let meta = self.evaluate(self.fresh_meta(rhs.1));
                    let x = UniqueName::new("x", rhs.1);
                    let body = self.bind(x.clone(), meta.clone()).fresh_meta(lhs.1);
                    let closure = self.closure(body);
                    let target = Rc::new(WithSpan(
                        Value::Pi {
                            name: x.clone(),
                            arg: meta.clone(),
                            body: closure.clone(),
                            implicit: *implicity,
                        },
                        lhs.1,
                    ));
                    self.try_unify(lhs_ty, target);
                    (meta, x, closure)
                };
                let rhs = self.check(rhs.clone(), arg_ty)?;
                let app = Rc::new(WithSpan(Term::App(lhs, rhs.clone(), *implicity), term.1));
                let ty = body(name, self.evaluate(rhs), self.env.global());
                Ok((app, ty))
            }
            Term::Universe => {
                let span = term.1;
                Ok((term, Rc::new(WithSpan(Value::Universe, span))))
            }
            Term::Pi {
                name,
                arg,
                body,
                implicit,
            } => {
                let arg = self.check_universe(arg.clone())?;
                let arg_value = self.evaluate(arg.clone());
                let body = self
                    .bind(name.clone(), arg_value)
                    .check_universe(body.clone())?;
                Ok((
                    Rc::new(WithSpan(
                        Term::Pi {
                            name: name.clone(),
                            arg,
                            body,
                            implicit: *implicit,
                        },
                        term.1,
                    )),
                    Rc::new(WithSpan(Value::Universe, term.1)),
                ))
            }

            Term::Let {
                name,
                ty,
                binding,
                body,
            } => {
                let ty = ty
                    .as_ref()
                    .map(|ty| self.check_universe(ty.clone()))
                    .unwrap_or_else(|| Ok(self.fresh_meta(name.span())))?;
                let vty = self.evaluate(ty.clone());
                let binding = self.check(binding.clone(), vty.clone())?;
                let vbinding = self.evaluate(binding.clone());
                let (body, term_ty) = self
                    .define(name.clone(), vbinding, vty)
                    .infer(body.clone())?;
                let term = Rc::new(WithSpan(
                    Term::Let {
                        name: name.clone(),
                        ty: Some(ty),
                        binding,
                        body,
                    },
                    term.1,
                ));
                Ok((term, term_ty))
            }

            Term::Hole => {
                let ty = self.evaluate(self.fresh_meta(term.1));
                Ok((self.fresh_meta(term.1), ty))
            }

            _ => todo!(),
        }
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::sema::term::test::*;
    #[test]
    fn it_checks_telescope_id() {
        _ = tracing_subscriber::fmt::try_init();
        let id_ty = pi("A", true, universe(), |a| arrow(a.clone(), a));
        let id = lam(["x"], |[x]| x);
        println!("{}", **id);
        let global = Context::new();
        let elab = ElabContext::new(&global);
        let ty = elab.evaluate(id_ty);
        let id = elab.check(id, ty).unwrap();
        println!("{}", **id);
    }

    #[test]
    fn it_checks_boolean() {
        _ = tracing_subscriber::fmt::try_init();
        let boolean = pi("B", false, hole(), |b| {
            arrow(b.clone(), arrow(b.clone(), b))
        }); // (B : _) -> B -> B -> B
        let r#true = lam(["B", "t", "f"], |[_, t, _]| t);
        let r#false = lam(["B", "t", "f"], |[_, _, f]| f);
        #[allow(non_snake_case)]
        let not = lam(["b", "B", "t", "f"], |[b, B, t, f]| app(b, [B, f, t]));

        let global = Context::new();
        let elab = ElabContext::new(&global);
        let ty = elab.evaluate(boolean);
        let checked = elab.check(r#true, ty.clone()).unwrap();
        println!("{}", **checked);
    }
}
