use either::Either::Left;
use rpds::{HashTrieMap, Vector};

use crate::{
    Result,
    eval::{Environment, quote},
    meta::MetaContext,
    term::{Term, TermPtr},
    utils::{DBLvl, Icit, Name, Pruning, Span, with_span},
    value::{Value, ValuePtr},
};
#[derive(Debug, Clone)]
pub struct Context {
    env: Environment,
    pub(crate) level: DBLvl,
    locals: Vector<(Name, VarKind)>,
    pub(crate) pruning: Pruning,
    name_types: HashTrieMap<Name, (DBLvl, ValuePtr)>,
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
            level: DBLvl(0),
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
        name: Name,
        ty: ValuePtr,
        f: F,
    ) -> Result<R>
    where
        F: FnOnce(&mut Self) -> Result<R>,
    {
        let old;
        if let Some(entry) = self.name_types.get_mut(&name) {
            old = Some(entry.clone());
            *entry = (self.level, ty.clone());
        } else {
            old = None;
            self.name_types.insert_mut(name, (self.level, ty.clone()));
        }
        let res = self.with_binder(meta, name, ty, f);
        if let Some(old) = old {
            self.name_types.insert_mut(name, old);
        } else {
            self.name_types.remove_mut(&name);
        }
        res
    }
    pub fn with_binder<F, R>(
        &mut self,
        meta: &MetaContext,
        name: Name,
        ty: ValuePtr,
        f: F,
    ) -> Result<R>
    where
        F: FnOnce(&mut Self) -> Result<R>,
    {
        let var = with_span(Value::var(self.level), name.span);
        self.env.push_back_mut(var.clone());
        let ty = match quote(self.level, ty, meta) {
            Ok(ty) => ty,
            Err(e) => {
                self.env.pop_back_mut();
                return Err(e);
            }
        };
        self.locals.push_back_mut((name, VarKind::Bound { ty }));
        self.pruning.push_back_mut(Some(Icit::Expl));
        let res = f(self);
        self.pruning.drop_last_mut();
        self.locals.drop_last_mut();
        self.env.pop_back_mut();
        res
    }
    pub fn with_def<F, R>(
        &mut self,
        name: Name,
        term: TermPtr,
        value: ValuePtr,
        ty: TermPtr,
        vty: ValuePtr,
        f: F,
    ) -> R
    where
        F: FnOnce(&mut Self) -> R,
    {
        let old;
        self.env.push_back_mut(value);
        self.locals.push_back_mut((
            name,
            VarKind::Defined {
                term: term.clone(),
                ty: ty.clone(),
            },
        ));
        if let Some(entry) = self.name_types.get_mut(&name) {
            old = Some(entry.clone());
            *entry = (self.level, vty.clone());
        } else {
            old = None;
            self.name_types.insert_mut(name, (self.level, vty.clone()));
        }
        let res = f(self);
        if let Some(old) = old {
            self.name_types.insert_mut(name, old);
        } else {
            self.name_types.remove_mut(&name);
        }
        self.locals.drop_last_mut();
        self.env.pop_back_mut();
        res
    }
    pub fn close_term(&self, term: TermPtr, span: Span) -> TermPtr {
        self.locals
            .iter()
            .fold(term, |body, (name, kind)| match kind {
                VarKind::Bound { .. } => {
                    with_span(Term::Lambda(*name, Left(Icit::Expl), None, body), span)
                }
                VarKind::Defined { term, ty } => with_span(
                    Term::Let {
                        name: *name,
                        ty: ty.clone(),
                        term: term.clone(),
                        body,
                    },
                    span,
                ),
            })
    }
    pub fn close_type(&self, ty: TermPtr, span: Span) -> TermPtr {
        self.locals
            .iter()
            .fold(ty, |body, (name, kind)| match kind {
                VarKind::Bound { ty } => {
                    with_span(Term::Pi(*name, Icit::Expl, ty.clone(), body), span)
                }
                VarKind::Defined { ty, term } => with_span(
                    Term::Let {
                        name: *name,
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
