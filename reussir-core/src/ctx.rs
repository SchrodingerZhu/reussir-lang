use rpds::{HashTrieMap, Vector};

use crate::utils::Closure;
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
    pub(crate) locals: Vector<(Name, VarKind)>,
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

    pub fn bind_mut(&mut self, meta: &MetaContext, name: Name, ty: ValuePtr) -> Result<()> {
        let quoted = quote(self.level, ty.clone(), meta)?;
        self.env
            .push_back_mut(with_span(Value::var(self.level), ty.span));
        self.pruning.push_back_mut(Some(Icit::Expl));
        self.name_types.insert_mut(name, (self.level, ty));
        self.locals
            .push_back_mut((name, VarKind::Bound { ty: quoted }));
        self.level = self.level.next();
        Ok(())
    }

    pub fn unbind_mut(&mut self, name: Name) {
        self.locals.drop_last_mut();
        self.pruning.drop_last_mut();
        self.env.pop_back_mut();
        self.level = DBLvl(self.level.0 - 1);
        self.name_types.remove_mut(&name);
    }

    pub fn binder_mut(&mut self, meta: &MetaContext, name: Name, ty: ValuePtr) -> Result<()> {
        let quoted = quote(self.level, ty.clone(), meta)?;
        self.env
            .push_back_mut(with_span(Value::var(self.level), ty.span));
        self.pruning.push_back_mut(Some(Icit::Expl));
        self.locals
            .push_back_mut((name, VarKind::Bound { ty: quoted }));
        self.level = self.level.next();
        Ok(())
    }

    pub fn unbinder_mut(&mut self) {
        self.locals.drop_last_mut();
        self.pruning.drop_last_mut();
        self.env.pop_back_mut();
        self.level = DBLvl(self.level.0 - 1);
    }

    pub fn def_mut(
        &mut self,
        name: Name,
        term: TermPtr,
        value: ValuePtr,
        ty: TermPtr,
        vty: ValuePtr,
    ) -> Result<Option<(DBLvl, ValuePtr)>> {
        let old;
        self.env.push_back_mut(value);
        self.locals.push_back_mut((
            name,
            VarKind::Defined {
                term: term.clone(),
                ty: ty.clone(),
            },
        ));
        self.pruning.push_back_mut(None);
        if let Some(entry) = self.name_types.get_mut(&name) {
            old = Some(entry.clone());
            *entry = (self.level, vty.clone());
        } else {
            old = None;
            self.name_types.insert_mut(name, (self.level, vty.clone()));
        }
        Ok(old)
    }
    pub fn undef_mut(&mut self, name: Name, old: Option<(DBLvl, ValuePtr)>) {
        if let Some(old) = old {
            self.name_types.insert_mut(name, old);
        } else {
            self.name_types.remove_mut(&name);
        }
        self.locals.drop_last_mut();
        self.env.pop_back_mut();
        self.pruning.drop_last_mut();
    }
    pub fn close_term(&self, term: TermPtr, span: Span) -> TermPtr {
        self.locals
            .iter()
            .rev()
            .fold(term, |body, (name, kind)| match kind {
                VarKind::Bound { .. } => with_span(Term::Lambda(*name, Icit::Expl, body), span),
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
            .rev()
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
    pub fn lookup(&self, name: Name) -> Option<&(DBLvl, ValuePtr)> {
        self.name_types.get(&name)
    }
    pub fn val_to_closure(&self, val: ValuePtr, meta: &MetaContext) -> Result<Closure> {
        let env = self.env.clone();
        let term = quote(self.level.next(), val, meta)?;
        Ok(Closure::new(env, term))
    }
}

#[derive(Debug)]
pub enum VarKind {
    Defined { term: TermPtr, ty: TermPtr },
    Bound { ty: TermPtr },
}
