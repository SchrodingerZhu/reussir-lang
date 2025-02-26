#![allow(unused)]
mod term;
use std::{
    marker::PhantomData,
    ops::{Deref, DerefMut},
};

use chumsky::span::SimpleSpan;
use gc_arena::{Arena, Collect, Gc, Mutation, Rootable, Static, allocator_api::MetricsAlloc, lock};
use rustc_hash::{FxBuildHasher, FxRandomState};
use smallvec::SmallVec;

use crate::syntax::{self, WithSpan};
use term::Term;
type Ref<'gc, T> = Gc<'gc, T>;
type RefCell<'gc, T> = Gc<'gc, lock::RefLock<T>>;
type Cell<'gc, T> = Gc<'gc, lock::Lock<T>>;
type HashMap<'gc, K, V> = hashbrown::HashMap<K, V, FxRandomState, MetricsAlloc<'gc>>;
type Vec<'gc, T> = allocator_api2::vec::Vec<T, MetricsAlloc<'gc>>;
type Box<'gc, T> = allocator_api2::boxed::Box<T, MetricsAlloc<'gc>>;
type UStr = Static<ustr::Ustr>;

#[derive(Debug, Clone, Copy, Eq, PartialEq, Hash, Collect)]
#[collect(require_static)]
pub enum FieldName {
    Idx(usize),
    Name(UStr),
}

#[derive(Debug, Clone, Eq, PartialEq, Hash, Collect)]
#[collect(no_drop)]
pub struct QualifiedName<'gc>(Box<'gc, [UStr]>, UStr);

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
#[derive(Copy, Clone, Collect, Eq)]
#[collect(no_drop)]
#[repr(transparent)]
pub struct UniqueName<'gc>(Gc<'gc, Static<WithSpan<ustr::Ustr>>>);

impl<'gc> UniqueName<'gc> {
    fn new<T: Into<ustr::Ustr>>(mc: &Mutation<'gc>, name: T, span: SimpleSpan) -> Self {
        Self(Gc::new(mc, Static(WithSpan(name.into(), span))))
    }
    fn fresh(mc: &Mutation<'gc>, span: SimpleSpan) -> Self {
        Self(Gc::new(mc, Static(WithSpan("$x".into(), span))))
    }
    fn span(&self) -> SimpleSpan {
        self.0.1
    }
    fn name(&self) -> ustr::Ustr {
        *self.0.0
    }
}

impl std::fmt::Debug for UniqueName<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}@{:?}", ***self.0, Gc::as_ptr(self.0))
    }
}

impl PartialEq for UniqueName<'_> {
    fn eq(&self, other: &Self) -> bool {
        Gc::ptr_eq(self.0, other.0)
    }
}

impl std::hash::Hash for UniqueName<'_> {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        Gc::as_ptr(self.0).hash(state);
    }
}

#[cfg(test)]
mod test {
    use super::*;
    #[test]
    fn it_creates_qualified_name() {
        _ = tracing_subscriber::fmt::try_init();
        let fake_name = syntax::QualifiedName::new(&["std", "test"], "test");
        let mut arena = Arena::<Rootable![CtxRef<'_>]>::new(|mc| Context::new(mc));
        arena.mutate(|mc, _ctx| {
            let name = QualifiedName::new(mc, fake_name);
        });
        arena.finish_cycle();
    }
}
