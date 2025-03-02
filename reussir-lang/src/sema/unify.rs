use std::{ops::ControlFlow, rc::Rc};

use rustc_hash::FxHashMapRand;

use thiserror::Error;

use crate::syntax::WithSpan;

use super::{
    Context, UniqueName,
    eval::{Environment, evaluate, force, value_apply},
    term::{Term, TermPtr},
    value::{Spine, SpineItem, Value, ValuePtr, empty_spine},
};

#[derive(Default)]
struct PartialRenaming {
    map: FxHashMapRand<UniqueName, UniqueName>,
    inversion: Vec<(UniqueName, bool)>,
}

impl std::ops::Deref for PartialRenaming {
    type Target = FxHashMapRand<UniqueName, UniqueName>;

    fn deref(&self) -> &Self::Target {
        &self.map
    }
}

#[derive(Error, Debug, Clone)]
pub enum Error {
    #[error("spine does not match pattern unification constraint")]
    InvalidPatternUnification,
    #[error("meta variable ?{0} occurs multiple times")]
    MultipleOccurence(usize),
    #[error("variable {} escapes its scope", .0.0.0)]
    EscapingVariable(UniqueName),
    #[error("failed to unify terms")]
    RigidMismatch,
}

impl PartialRenaming {
    fn new() -> Self {
        Self::default()
    }
    fn locally<F, R>(&mut self, name: UniqueName, f: F) -> R
    where
        F: for<'a> FnOnce(&'a mut Self, UniqueName) -> R,
    {
        let fresh = name.refresh();
        {
            let replaced = self.map.insert(name.clone(), fresh.clone());
            assert!(replaced.is_none());
        }
        let res = f(self, fresh);
        {
            let removed = self.map.remove(&name);
            assert!(removed.is_some());
        }
        res
    }
    fn try_invert(spine: &Spine, global: &Context) -> Result<Self, Error> {
        spine
            .iter()
            .cloned()
            .try_fold(Self::default(), |mut acc, x| {
                match &**force(x.value, global) {
                    Value::Rigid(name, spine) if spine.is_empty() && !acc.contains_key(name) => {
                        acc.map.insert(name.clone(), name.refresh());
                        acc.inversion.push((name.clone(), x.implicit));
                        ControlFlow::Continue(acc)
                    }
                    _ => ControlFlow::Break(()),
                }
            })
            .continue_value()
            .ok_or(Error::InvalidPatternUnification)
    }
    fn rename(&mut self, global: &Context, meta: usize, value: ValuePtr) -> Result<TermPtr, Error> {
        let span = value.1;

        let mut rename_spine = |renaming: &mut PartialRenaming,
                                term: TermPtr,
                                spine: &Spine|
         -> Result<TermPtr, Error> {
            spine.iter().try_fold(term, |acc, x| {
                let arg = renaming.rename(global, meta, x.value.clone())?;
                let term = Term::App(acc, arg, x.implicit);
                Ok(Rc::new(WithSpan(term, span)))
            })
        };

        match &**force(value, global) {
            Value::Stuck(tm) => Ok(tm.clone()),
            Value::Rigid(var, spine) => {
                let renamed = self.get(var).cloned();
                match renamed {
                    None => Err(Error::EscapingVariable(var.clone())),
                    Some(fresh) => {
                        let term = Rc::new(WithSpan(Term::Var(fresh), span));
                        rename_spine(self, term, spine)
                    }
                }
            }
            Value::Flex(m, _) if *m == meta => Err(Error::MultipleOccurence(*m)),
            Value::Flex(m, spine) => {
                let term = Rc::new(WithSpan(Term::Meta(*m), span));
                rename_spine(self, term, spine)
            }
            Value::Universe => Ok(Rc::new(WithSpan(Term::Universe, span))),
            Value::Pi {
                name,
                arg,
                body,
                implicit,
            } => {
                let arg = self.rename(global, meta, arg.clone())?;
                self.locally(name.clone(), |renaming, renamed| {
                    let var = Rc::new(WithSpan(Value::Rigid(renamed, empty_spine()), name.span()));
                    let body = renaming.rename(global, meta, body(name.clone(), var, global))?;
                    Ok(Rc::new(WithSpan(
                        Term::Pi {
                            name: name.clone(),
                            arg,
                            body,
                            implicit: *implicit,
                        },
                        span,
                    )))
                })
            }
            Value::Lambda {
                name,
                body,
                implicit,
            } => self.locally(name.clone(), |renaming, renamed| {
                let var = Rc::new(WithSpan(Value::Rigid(renamed, empty_spine()), name.span()));
                let body = renaming.rename(global, meta, body(name.clone(), var, global))?;
                Ok(Rc::new(WithSpan(
                    Term::Lambda {
                        binding: name.clone(),
                        body,
                        implicit: *implicit,
                    },
                    span,
                )))
            }),
            Value::Invalid => Ok(Rc::new(WithSpan(Term::Invalid, span))),
        }
    }

    fn solution(self, body: TermPtr) -> TermPtr {
        let span = body.1;
        self.inversion
            .into_iter()
            .fold(body, |body, (binding, implicit)| {
                Rc::new(WithSpan(
                    Term::Lambda {
                        binding,
                        body,
                        implicit,
                    },
                    span,
                ))
            })
    }
}

fn solve_meta(meta: usize, spine: &Spine, rhs: ValuePtr, global: &Context) -> Result<(), Error> {
    let mut renaming = PartialRenaming::try_invert(spine, global)?;
    let rhs = renaming.rename(global, meta, rhs)?;
    let solution = renaming.solution(rhs);
    let empty = Environment::new(global);
    let value = evaluate(empty, solution);
    global.insert_meta(meta, value);
    Ok(())
}

fn unify(lhs: ValuePtr, rhs: ValuePtr, global: &Context) -> Result<(), Error> {
    let unify = |lhs, rhs| {
        stacker::maybe_grow(32 * 1024, 1024 * 1024, || {
            crate::sema::unify::unify(lhs, rhs, global)
        })
    };
    let lhs = force(lhs, global);
    let rhs = force(rhs, global);
    let get_fresh_var = |n: &UniqueName| {
        let fresh = n.refresh();
        Rc::new(WithSpan(
            Value::Rigid(fresh.clone(), empty_spine()),
            fresh.span(),
        ))
    };
    let unify_spine = |x: &Spine, y: &Spine| {
        if x.len() != y.len() {
            return Err(Error::RigidMismatch);
        }
        for (x, y) in x.iter().cloned().zip(y.iter().cloned()) {
            unify(x.value, y.value)?;
        }
        Ok(())
    };
    match (&**lhs, &**rhs) {
        (
            Value::Lambda {
                name: n0, body: b0, ..
            },
            Value::Lambda {
                name: n1, body: b1, ..
            },
        ) => {
            let var = get_fresh_var(n0);
            let lhs = b0(n0.clone(), var.clone(), global);
            let rhs = b1(n1.clone(), var, global);
            unify(lhs, rhs)
        }
        // eta expansion
        (
            _,
            Value::Lambda {
                name: n,
                body: b,
                implicit,
            },
        ) => {
            let var = get_fresh_var(n);
            let lhs = value_apply(
                lhs.clone(),
                SpineItem::new(var.clone(), *implicit),
                lhs.1,
                global,
            );
            let rhs = b(n.clone(), var, global);
            unify(lhs, rhs)
        }
        (
            Value::Lambda {
                name: n,
                body: b,
                implicit,
            },
            _,
        ) => {
            let var = get_fresh_var(n);
            let rhs = value_apply(
                rhs.clone(),
                SpineItem::new(var.clone(), *implicit),
                rhs.1,
                global,
            );
            let lhs = b(n.clone(), var, global);
            unify(lhs, rhs)
        }
        (Value::Stuck(x), Value::Stuck(y)) if x.is_alpha_equivalent(y) => Ok(()),
        (Value::Universe, Value::Universe) => Ok(()),
        (
            Value::Pi {
                name: n0,
                arg: a0,
                body: b0,
                implicit: i0,
            },
            Value::Pi {
                name: n1,
                arg: a1,
                body: b1,
                implicit: i1,
            },
        ) if i0 == i1 => {
            unify(a0.clone(), a1.clone())?;
            let var = get_fresh_var(n0);
            let lhs = b0(n0.clone(), var.clone(), global);
            let rhs = b1(n1.clone(), var, global);
            unify(lhs, rhs)
        }
        (Value::Rigid(n0, spine0), Value::Rigid(n1, spine1)) if n0 == n1 => {
            unify_spine(spine0, spine1)
        }
        (Value::Flex(m0, spine0), Value::Flex(m1, spine1)) if m0 == m1 => {
            unify_spine(spine0, spine1)
        }
        // unification
        (Value::Flex(m, spine), _) => solve_meta(*m, spine, rhs, global),
        (_, Value::Flex(m, spine)) => solve_meta(*m, spine, lhs, global),
        _ => Err(Error::RigidMismatch),
    }
}

#[cfg(test)]
mod test {
    use chumsky::span::SimpleSpan;
    use rpds::Queue;

    use super::*;
    use crate::sema::{eval::quote, term::test::*};

    #[test]
    fn it_checks_simple_equivalence() {
        let global = Context::new();
        let env = Environment::new(&global);
        let idx = evaluate(env.clone(), lam(["x"], |[x]| x));
        let idy = evaluate(env.clone(), lam(["y"], |[y]| y));
        unify(idx, idy, &global).unwrap();
    }

    #[test]
    fn it_checks_equivalence_after_eta() {
        let global = Context::new();
        let env = Environment::new(&global);
        let id = lam(["x"], |[x]| x);
        let id_plain = evaluate(env.clone(), id.clone());
        let id_eta = evaluate(env.clone(), lam(["x"], move |[x]| app(id, [x])));
        unify(id_plain, id_eta, &global).unwrap();
    }

    #[test]
    fn it_unifies_untyped_ids() {
        let global = Context::new();
        let env = Environment::new(&global);
        let id = lam(["A", "x"], |[_, x]| x);
        let fake = SimpleSpan::new(0, 0);
        let id_with_hole = lam(["A", "x"], |[a, x]| {
            let Term::Var(a_name) = &a.0 else {
                unreachable!()
            };
            let Term::Var(x_name) = &x.0 else {
                unreachable!()
            };
            let a_name = a_name.clone();
            let x_name = x_name.clone();
            let meta = global.fresh_meta(fake);
            let inserted_meta = Rc::new(WithSpan(
                Term::InsertedMeta(meta, Queue::new().enqueue(a_name).enqueue(x_name)),
                fake,
            ));
            app(id.clone(), [inserted_meta, x])
        });
        let id = evaluate(env.clone(), id.clone());
        let id_with_hole = evaluate(env.clone(), id_with_hole.clone());
        unify(id, id_with_hole.clone(), &global).unwrap();
        let term = quote(id_with_hole, &global);
        println!("{}", **term);
    }
}
