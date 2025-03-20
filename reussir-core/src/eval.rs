use archery::RcK;
use rpds::HashTrieMap;
use rustc_hash::FxRandomState;

use crate::{
    Error, Result,
    meta::MetaContext,
    term::TermPtr,
    utils::{Closure, Icit, Pruning, Spine, UniqueName, with_span_as},
    value::{Value, ValuePtr},
};

#[derive(Clone)]
pub struct Environment(HashTrieMap<UniqueName, ValuePtr, RcK, FxRandomState>);

impl std::fmt::Debug for Environment {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_map().entries(self.0.iter()).finish()
    }
}

impl Default for Environment {
    fn default() -> Self {
        Self::new()
    }
}

impl Environment {
    pub fn new() -> Self {
        Self(HashTrieMap::new_with_hasher_and_ptr_kind(
            FxRandomState::default(),
        ))
    }
    pub fn with_var<F, R>(&mut self, name: UniqueName, value: ValuePtr, f: F) -> R
    where
        F: FnOnce(&mut Environment) -> R,
    {
        self.0.insert_mut(name.clone(), value.clone());
        let res = f(self);
        self.0.remove_mut(&name);
        res
    }
    pub fn insert_mut(&mut self, name: UniqueName, value: ValuePtr) {
        self.0.insert_mut(name, value);
    }
    pub fn remove_mut(&mut self, name: &UniqueName) {
        self.0.remove_mut(name);
    }
    pub fn insert(&self, name: UniqueName, value: ValuePtr) -> Self {
        Self(self.0.insert(name, value))
    }
    pub fn get_var(&self, name: &UniqueName) -> Result<ValuePtr> {
        self.0
            .get(name)
            .cloned()
            .ok_or_else(|| Error::unresolved_variable(name.clone()))
    }
    pub fn evaluate(&mut self, term: TermPtr, meta: &MetaContext) -> Result<ValuePtr> {
        match term.data() {
            crate::term::Term::Hole => Err(Error::internal("Cannot evaluate hole")),
            crate::term::Term::Var(name) => self.get_var(name),
            crate::term::Term::Lambda(name, icit, body) => {
                let closure = Closure::new(self.clone(), body.clone());
                Ok(with_span_as(
                    Value::Lambda(name.clone(), *icit, closure),
                    term,
                ))
            }
            crate::term::Term::App(lhs, rhs, icit) => {
                let lhs = self.evaluate(lhs.clone(), meta)?;
                let rhs = self.evaluate(rhs.clone(), meta)?;
                app_val(lhs, rhs, *icit, meta)
            }
            crate::term::Term::AppPruning(term, pruning) => {
                let term = self.evaluate(term.clone(), meta)?;
                self.app_pruning(term, pruning, meta)
            }
            crate::term::Term::Universe => Ok(with_span_as(Value::Universe, term)),
            crate::term::Term::Pi(name, icit, ty, body) => {
                let closure = Closure::new(self.clone(), body.clone());
                let ty = self.evaluate(ty.clone(), meta)?;
                Ok(with_span_as(
                    Value::Pi(name.clone(), *icit, ty, closure),
                    term,
                ))
            }
            crate::term::Term::Let {
                name, term, body, ..
            } => {
                let term = self.evaluate(term.clone(), meta)?;
                self.with_var(name.clone(), term, |env| env.evaluate(body.clone(), meta))
            }
            crate::term::Term::Meta(m) => meta.get_meta_value(*m, term.span),
            crate::term::Term::Postponed(c) => meta.get_check_value(self, *c, term.span),
        }
    }
    pub fn app_pruning(
        &self,
        value: ValuePtr,
        pruning: &Pruning,
        meta: &MetaContext,
    ) -> Result<ValuePtr> {
        pruning.iter().try_fold(value, |acc, (name, icit)| {
            let var = self.0.get(name).ok_or_else(|| {
                Error::internal(format!("Variable {:?} not found in environment", name))
            })?;
            app_val(acc, var.clone(), *icit, meta)
        })
    }
}

fn app_val(lhs: ValuePtr, rhs: ValuePtr, icit: Icit, meta: &MetaContext) -> Result<ValuePtr> {
    match lhs.data() {
        Value::Lambda(name, _, closure) => closure.apply(name.clone(), rhs, meta),
        Value::Flex(meta, spine) => Ok(with_span_as(
            Value::Flex(*meta, spine.push_back((rhs, icit))),
            lhs,
        )),
        Value::Rigid(name, spine) => Ok(with_span_as(
            Value::Rigid(name.clone(), spine.push_back((rhs, icit))),
            lhs,
        )),
        _ => Err(Error::internal(format!(
            "Cannot apply {:?} to {:?}",
            lhs, rhs
        ))),
    }
}

fn app_spine(value: ValuePtr, spine: &Spine, meta: &MetaContext) -> Result<ValuePtr> {
    spine
        .iter()
        .cloned()
        .try_fold(value, |acc, (arg, icit)| app_val(acc, arg, icit, meta))
}

pub fn quote(value: ValuePtr, global: &MetaContext) -> TermPtr {
    todo!()
}
