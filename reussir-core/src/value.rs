use std::rc::Rc;

use crate::{
    meta::MetaVar,
    utils::{Closure, Icit, Spine, UniqueName, WithSpan, empty_spine},
};

pub type ValuePtr = Rc<WithSpan<Value>>;

#[derive(Debug, Clone)]
pub enum Value {
    Flex(MetaVar, Spine),
    Rigid(UniqueName, Spine),
    Lambda(UniqueName, Icit, Closure),
    Pi(UniqueName, Icit, ValuePtr, Closure),
    Universe,
}

impl Value {
    pub fn var(name: UniqueName) -> Self {
        Self::Rigid(name, empty_spine())
    }
    pub fn meta(meta: MetaVar) -> Self {
        Self::Flex(meta, empty_spine())
    }
}
