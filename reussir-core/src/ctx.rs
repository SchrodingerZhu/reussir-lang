use rpds::{HashTrieMap, Vector};

use crate::{
    eval::{Environment, quote},
    meta::MetaContext,
    term::TermPtr,
    utils::{Icit, Pruning, UniqueName, with_span},
    value::{Value, ValuePtr},
};
#[derive(Debug, Clone)]
pub struct Context {
    env: Environment,
    locals: Vector<(UniqueName, VarKind)>,
    pub(crate) pruning: Pruning,
    name_types: HashTrieMap<UniqueName, ValuePtr>,
}

impl Default for Context {
    fn default() -> Self {
        Self::new()
    }
}

impl Context {
    pub fn new() -> Self {
        Self {
            env: Environment::new(),
            locals: Vector::new(),
            pruning: Pruning::new(),
            name_types: HashTrieMap::new(),
        }
    }
    pub fn with_bind<F, R>(&mut self, meta: &MetaContext, name: UniqueName, ty: ValuePtr, f: F) -> R
    where
        F: FnOnce(&mut Self) -> R,
    {
        self.name_types.insert_mut(name.clone(), ty.clone());
        let res = self.with_binder(meta, name.clone(), ty, f);
        self.name_types.remove_mut(&name);
        res
    }
    pub fn with_binder<F, R>(
        &mut self,
        meta: &MetaContext,
        name: UniqueName,
        ty: ValuePtr,
        f: F,
    ) -> R
    where
        F: FnOnce(&mut Self) -> R,
    {
        let var = with_span(Value::var(name.clone()), name.start(), name.end());
        self.env.insert_mut(name.clone(), var.clone());
        self.locals.push_back_mut((
            name.clone(),
            VarKind::Bound {
                ty: quote(ty, meta),
            },
        ));
        self.pruning.push_back_mut((name.clone(), Icit::Expl));
        let res = f(self);
        self.pruning.drop_last_mut();
        self.locals.drop_last_mut();
        self.env.remove_mut(&name);
        res
    }
    pub fn with_def<F, R>(
        &mut self,
        name: UniqueName,
        term: TermPtr,
        value: ValuePtr,
        ty: TermPtr,
        vty: ValuePtr,
        f: F,
    ) -> R
    where
        F: FnOnce(&mut Self) -> R,
    {
        self.env.insert_mut(name.clone(), value);
        self.locals.push_back_mut((
            name.clone(),
            VarKind::Defined {
                term: term.clone(),
                ty: ty.clone(),
            },
        ));
        self.name_types.insert_mut(name.clone(), vty);
        let res = f(self);
        self.name_types.remove_mut(&name);
        self.locals.drop_last_mut();
        self.env.remove_mut(&name);
        res
    }
}

#[derive(Debug)]
pub enum VarKind {
    Defined { term: TermPtr, ty: TermPtr },
    Bound { ty: TermPtr },
}
