use std::rc::Rc;

use archery::RcK;
use chumsky::span::SimpleSpan;
use rpds::{HashTrieMap, Queue};
use rustc_hash::FxRandomState;

use crate::syntax::WithSpan;

use super::{
    Context, MetaEntry, UniqueName,
    eval::{Environment, value_apply},
    term::{Term, TermPtr},
};

pub type Map<K, V> = HashTrieMap<K, V, RcK, FxRandomState>;
pub type ValuePtr = Rc<WithSpan<Value>>;
pub type Closure = dyn for<'glb> Fn(UniqueName, ValuePtr, &'glb Context) -> ValuePtr;
pub type Spine = Queue<SpineItem>;

#[derive(Clone)]
pub enum Value {
    /// Stuck due to effectful terms
    Stuck(TermPtr),
    /// Stuck on variable
    Rigid(UniqueName, Spine),
    /// Stuck on unsolved meta
    Flex(usize, Spine),
    Universe,
    Pi {
        name: UniqueName,
        arg: ValuePtr,
        body: Rc<Closure>,
        implicit: bool,
    },
    Lambda {
        name: UniqueName,
        body: Rc<Closure>,
        implicit: bool,
    },
    Invalid,
}

#[derive(Clone)]
pub struct SpineItem {
    pub value: ValuePtr,
    pub implicit: bool,
}

impl SpineItem {
    pub fn new(value: ValuePtr, implicit: bool) -> Self {
        SpineItem { value, implicit }
    }
}

thread_local! {
    static EMPTY_SPINE : Spine = Spine::new();
}

pub(crate) fn empty_spine() -> Spine {
    EMPTY_SPINE.with(Clone::clone)
}
