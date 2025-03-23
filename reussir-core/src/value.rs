use std::rc::Rc;

use crate::{
    meta::MetaVar,
    utils::{empty_spine, Closure, DBLvl, Icit, Name, Spine, WithSpan},
};

pub type ValuePtr = Rc<WithSpan<Value>>;

#[derive(Debug, Clone)]
pub enum Value {
    Flex(MetaVar, Spine),
    Rigid(DBLvl, Spine),
    Lambda(Name, Icit, Closure),
    Pi(Name, Icit, ValuePtr, Closure),
    Universe,
}

impl Value {
    pub fn var(level: DBLvl) -> Self {
        Self::Rigid(level, empty_spine())
    }
    pub fn meta(meta: MetaVar) -> Self {
        Self::Flex(meta, empty_spine())
    }
}
