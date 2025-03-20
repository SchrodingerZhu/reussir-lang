use std::cell::RefCell;

use tinyset::SetUsize;

use crate::{
    Error, Result,
    ctx::Context,
    eval::{Environment, app_spine},
    term::TermPtr,
    utils::with_span,
    value::{Value, ValuePtr},
};

#[repr(transparent)]
#[derive(Debug, Clone, Copy, Eq, PartialEq, Hash)]
pub struct CheckVar(pub(crate) usize);

#[repr(transparent)]
#[derive(Debug, Clone, Copy, Eq, PartialEq, Hash)]
pub struct MetaVar(usize);

#[derive(Debug, Clone)]
pub enum CheckEntry {
    Checked(TermPtr),
    Unchecked {
        ctx: Context,
        term: TermPtr,
        ty: ValuePtr,
        meta: MetaVar,
    },
}

#[derive(Debug, Clone)]
pub enum MetaEntry {
    Unsolved { blocking: SetUsize, ty: ValuePtr },
    Solved { val: ValuePtr, ty: ValuePtr },
}

#[derive(Debug)]
pub struct MetaContext {
    checks: Vec<CheckEntry>,
    metas: Vec<MetaEntry>,
}

impl Default for MetaContext {
    fn default() -> Self {
        Self::new()
    }
}

impl MetaContext {
    pub fn new() -> Self {
        Self {
            checks: Vec::new(),
            metas: Vec::new(),
        }
    }
    pub fn new_check(
        &mut self,
        ctx: Context,
        term: TermPtr,
        ty: ValuePtr,
        meta: MetaVar,
    ) -> CheckVar {
        let checks = &mut self.checks;
        let var = CheckVar(checks.len());
        checks.push(CheckEntry::Unchecked {
            ctx,
            term,
            ty,
            meta,
        });
        var
    }
    pub fn get_meta_value(&self, var: MetaVar, span: (usize, usize)) -> Result<ValuePtr> {
        let metas = &self.metas;
        match metas.get(var.0) {
            Some(MetaEntry::Solved { val, .. }) => Ok(val.clone()),
            Some(MetaEntry::Unsolved { .. }) => Ok(with_span(Value::meta(var), span)),
            None => Err(Error::internal("invalid meta variable")),
        }
    }
    pub fn get_check_value(
        &self,
        env: &mut Environment,
        var: CheckVar,
        span: (usize, usize),
    ) -> Result<ValuePtr> {
        let checks = &self.checks;
        match checks.get(var.0) {
            Some(CheckEntry::Checked(term)) => env.evaluate(term.clone(), self),
            Some(CheckEntry::Unchecked { ctx, meta, .. }) => {
                env.app_pruning(self.get_meta_value(*meta, span)?, &ctx.pruning, self, span)
            }
            None => Err(Error::internal("invalid check variable")),
        }
    }
    pub fn get_check<F, R>(&self, var: CheckVar) -> Result<CheckEntry> {
        let checks = &self.checks;
        checks
            .get(var.0)
            .cloned()
            .ok_or_else(|| Error::internal("invalid check variable"))
    }
    pub fn modify_check<F>(&mut self, var: CheckVar, conti: F) -> Result<()>
    where
        F: FnOnce(&mut CheckEntry) -> Result<()>,
    {
        let checks = &mut self.checks;
        match checks.get_mut(var.0) {
            Some(entry) => conti(entry),
            None => Err(Error::internal("invalid check variable")),
        }
    }
    pub fn new_meta(&mut self, ty: ValuePtr) -> MetaVar {
        let metas = &mut self.metas;
        let var = MetaVar(metas.len());
        metas.push(MetaEntry::Unsolved {
            blocking: SetUsize::new(),
            ty,
        });
        var
    }
    pub fn set_meta(&mut self, var: MetaVar, new_entry: MetaEntry) -> Result<()> {
        let metas = &mut self.metas;
        match metas.get_mut(var.0) {
            Some(entry) => {
                *entry = new_entry;
                Ok(())
            }
            None => Err(Error::internal("invalid meta variable")),
        }
    }
    pub fn get_meta(&self, var: MetaVar) -> Result<&MetaEntry> {
        let metas = &self.metas;
        match metas.get(var.0) {
            Some(entry) => Ok(&entry),
            None => Err(Error::unresolved_meta(var)),
        }
    }
    pub fn add_blocker(&mut self, chk: CheckVar, meta: MetaVar) -> Result<()> {
        let metas = &mut self.metas;
        match metas.get_mut(meta.0) {
            Some(MetaEntry::Unsolved { blocking, .. }) => {
                blocking.insert(chk.0);
                Ok(())
            }
            Some(MetaEntry::Solved { .. }) => Err(Error::internal("meta variable already solved")),
            None => Err(Error::internal("invalid meta variable")),
        }
    }
    pub fn force(&self, val: ValuePtr) -> Result<ValuePtr> {
        match val.data() {
            Value::Flex(m, sp) => match self.metas.get(m.0) {
                Some(MetaEntry::Unsolved { .. }) => Ok(val),
                Some(MetaEntry::Solved { val, .. }) => {
                    self.force(app_spine(val.clone(), sp, self, val.span)?)
                }
                None => Err(Error::internal("invalid meta variable")),
            },
            _ => Ok(val),
        }
    }
}
