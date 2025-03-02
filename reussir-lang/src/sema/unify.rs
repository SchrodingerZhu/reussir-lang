use std::{ops::ControlFlow, rc::Rc};

use rustc_hash::FxHashMapRand;

use thiserror::Error;

use crate::syntax::WithSpan;

use super::{
    Context, UniqueName,
    eval::force,
    term::{Term, TermPtr},
    value::{Spine, SpineItem, Value, ValuePtr, empty_spine},
};

#[repr(transparent)]
#[derive(Default)]
struct PartialRenaming(FxHashMapRand<UniqueName, UniqueName>);

impl std::ops::Deref for PartialRenaming {
    type Target = FxHashMapRand<UniqueName, UniqueName>;

    fn deref(&self) -> &Self::Target {
        &self.0
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
}

impl PartialRenaming {
    fn new() -> Self {
        Self::default()
    }
    fn with<F, R>(&mut self, name: UniqueName, f: F) -> R
    where
        F: for<'a> FnOnce(&'a mut Self, UniqueName) -> R,
    {
        let fresh = name.refresh();
        {
            let replaced = self.0.insert(name.clone(), fresh.clone());
            assert!(replaced.is_none());
        }
        let res = f(self, fresh);
        {
            let removed = self.0.remove(&name);
            assert!(removed.is_some());
        }
        res
    }
    fn try_invert(spine: &Spine, global: &Context) -> Result<Self, Error> {
        spine
            .iter()
            .cloned()
            .map(|x| x.value)
            .try_fold(FxHashMapRand::default(), |mut acc, v| {
                match &**force(v, global) {
                    Value::Rigid(name, spine) if spine.is_empty() && !acc.contains_key(name) => {
                        acc.insert(name.clone(), name.refresh());
                        ControlFlow::Continue(acc)
                    }
                    _ => ControlFlow::Break(()),
                }
            })
            .continue_value()
            .map(Self)
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
                self.with(name.clone(), |renaming, renamed| {
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
            } => self.with(name.clone(), |renaming, renamed| {
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
}
