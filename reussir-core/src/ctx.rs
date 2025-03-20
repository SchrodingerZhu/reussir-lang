use rpds::{HashTrieMap, Vector};

use crate::{
    Result,
    eval::{Environment, quote},
    meta::MetaContext,
    term::{Term, TermPtr},
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
    pub fn env_mut(&mut self) -> &mut Environment {
        &mut self.env
    }
    pub fn env_mut_pruning(&mut self) -> (&mut Environment, &Pruning) {
        (&mut self.env, &self.pruning)
    }
    pub fn with_bind<F, R>(
        &mut self,
        meta: &MetaContext,
        name: UniqueName,
        ty: ValuePtr,
        f: F,
    ) -> Result<R>
    where
        F: FnOnce(&mut Self) -> Result<R>,
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
    ) -> Result<R>
    where
        F: FnOnce(&mut Self) -> Result<R>,
    {
        let var = with_span(Value::var(name.clone()), name.span());
        self.env.insert_mut(name.clone(), var.clone());
        let ty = match quote(ty, meta) {
            Ok(ty) => ty,
            Err(e) => {
                self.env.remove_mut(&name);
                return Err(e);
            }
        };
        self.locals
            .push_back_mut((name.clone(), VarKind::Bound { ty }));
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
    pub fn close_term(&self, term: TermPtr, span: (usize, usize)) -> TermPtr {
        self.locals
            .iter()
            .fold(term, |body, (name, kind)| match kind {
                VarKind::Bound { .. } => {
                    with_span(Term::Lambda(name.clone(), Icit::Expl, body), span)
                }
                VarKind::Defined { term, ty } => with_span(
                    Term::Let {
                        name: name.clone(),
                        ty: ty.clone(),
                        term: term.clone(),
                        body,
                    },
                    span,
                ),
            })
    }
    pub fn close_type(&self, ty: TermPtr, span: (usize, usize)) -> TermPtr {
        self.locals
            .iter()
            .fold(ty, |body, (name, kind)| match kind {
                VarKind::Bound { ty } => {
                    with_span(Term::Pi(name.clone(), Icit::Expl, ty.clone(), body), span)
                }
                VarKind::Defined { ty, term } => with_span(
                    Term::Let {
                        name: name.clone(),
                        ty: ty.clone(),
                        term: term.clone(),
                        body,
                    },
                    span,
                ),
            })
    }
}

#[derive(Debug)]
pub enum VarKind {
    Defined { term: TermPtr, ty: TermPtr },
    Bound { ty: TermPtr },
}
