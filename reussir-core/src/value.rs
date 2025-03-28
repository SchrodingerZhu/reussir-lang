use std::rc::Rc;

use crate::{
    meta::MetaVar,
    utils::{Closure, DBLvl, Icit, Name, Spine, WithSpan, empty_spine},
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

impl std::fmt::Display for Value {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Value::Flex(meta, spine) => write!(f, "?{}[{}]", meta.0, spine.len()),
            Value::Rigid(level, spine) => write!(f, "{}[{}]", level, spine.len()),
            Value::Lambda(name, Icit::Impl, c) => {
                write!(f, "λ{{{}}} . {}",  name.data(), c)
            }
            Value::Lambda(name, Icit::Expl, c) => {
                write!(f, "λ{} . {}", name.data(), c)
            }
            Value::Pi(name, Icit::Impl, ty, c) if !name.is_anon() => {
                write!(f, "Π {{{} : {}}}.{})", name.data(), ty.data(), c)
            }
            Value::Pi(_, Icit::Impl, ty, c) => {
                write!(f, "{} -> {}", ty.data(), c)
            }
            Value::Pi(name, Icit::Expl, ty, c) if !name.is_anon() => {
                write!(f, "Π ({} : {}) . {}", name.data(), ty.data(), c)
            }
            Value::Pi(_, Icit::Expl, ty, c) => {
                write!(f, "{} -> {}", ty.data(), c)
            }
            Value::Universe => write!(f, "𝓤"),
        }
    }
}