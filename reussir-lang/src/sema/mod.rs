#![allow(unused)]
mod term;
mod r#type;
use std::{
    marker::PhantomData,
    ops::{Deref, DerefMut},
};

use allocator_api2::{boxed::Box, vec::Vec};
use gc_arena::{allocator_api::MetricsAlloc, lock, Arena, Collect, Gc, Mutation, Rootable, Static};
use rustc_hash::{FxBuildHasher, FxRandomState};
use smallvec::SmallVec;

use crate::syntax;

type Ref<'gc, T> = Gc<'gc, T>;
type RefCell<'gc, T> = Gc<'gc, lock::RefLock<T>>;
type Cell<'gc, T> = Gc<'gc, lock::Lock<T>>;
type HashMap<'gc, K, V> = hashbrown::HashMap<K, V, FxRandomState, MetricsAlloc<'gc>>;
type UStr = Static<ustr::Ustr>;

#[derive(Debug, Clone, Eq, PartialEq, Hash, Collect)]
#[collect(no_drop)]
pub struct QualifiedName<'gc>(Box<[UStr], MetricsAlloc<'gc>>, UStr);

impl<'gc> QualifiedName<'gc> {
    pub fn new<'a>(mutator: &Mutation<'gc>, name: syntax::QualifiedName<'a>) -> Self {
        let allocator = MetricsAlloc::new(mutator);
        let mut vec = Vec::with_capacity_in(name.qualifier().len(), allocator);
        vec.extend(
            name.qualifier()
                .iter()
                .copied()
                .map(ustr::ustr)
                .map(UStr::from),
        );
        let basename = ustr::Ustr::from(name.basename()).into();
        Self(vec.into_boxed_slice(), basename)
    }

    pub fn qualifier(&self) -> &[UStr] {
        &self.0
    }

    pub fn basename(&self) -> &UStr {
        &self.1
    }
}

#[derive(Clone, Collect)]
#[collect(no_drop)]
pub struct Context<'gc> {
    functions: HashMap<'gc, QualifiedName<'gc>, Ref<'gc, Term<'gc>>>,
}

impl<'gc> Context<'gc> {
    pub fn new(mutator: &Mutation<'gc>) -> Ref<'gc, Self> {
        let alloc = MetricsAlloc::new(mutator);
        let functions = HashMap::with_hasher_in(FxRandomState::new(), alloc);
        Gc::new(mutator, Self { functions })
    }
}

type CtxRef<'gc> = Ref<'gc, Context<'gc>>;

#[derive(Clone, Collect)]
#[collect(no_drop)]
enum Term<'gc> {
    Todo(PhantomData<&'gc ()>),
}

#[cfg(test)]
mod test {
    use super::*;
    #[test]
    fn it_creates_qualified_name() {
        tracing_subscriber::fmt::init();
        let fake_name = syntax::QualifiedName::new(&["std", "test"], "test");
        let mut arena = Arena::<Rootable![CtxRef<'_>]>::new(|mc| Context::new(mc));
        arena.mutate(|mc, _ctx| {
            let name = QualifiedName::new(mc, fake_name);
        });
        arena.finish_cycle();
    }
}
