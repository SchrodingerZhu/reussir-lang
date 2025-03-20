use archery::RcK;
use rpds::HashTrieMap;
use rustc_hash::FxRandomState;

use crate::{
    meta::MetaContext,
    term::TermPtr,
    utils::{with_span, Icit, Spine, UniqueName},
    value::{Value, ValuePtr},
    Error, Result,
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
    pub fn evaluate(&mut self, term: TermPtr, meta: &MetaContext) -> Result<ValuePtr> {
        todo!()
    }
}

fn app_val(lhs: ValuePtr, rhs: ValuePtr, icit: Icit, meta: &MetaContext) -> Result<ValuePtr> {
    let span_min = lhs.start.min(rhs.start);
    let span_max = lhs.end.max(rhs.end);
    match lhs.data() {
        Value::Lambda(name, _, closure) => closure.apply(name.clone(), rhs, meta),
        Value::Flex(meta, spine) => Ok(with_span(
            Value::Flex(meta.clone(), spine.push_back((rhs, icit))),
            span_min,
            span_max,
        )),
        Value::Rigid(name, spine) => Ok(with_span(
            Value::Rigid(name.clone(), spine.push_back((rhs, icit))),
            span_min,
            span_max,
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
