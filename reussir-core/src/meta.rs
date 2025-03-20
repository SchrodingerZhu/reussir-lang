use std::cell::RefCell;

use tinyset::SetUsize;

use crate::{
    Error, Result,
    ctx::Context,
    eval::Environment,
    term::TermPtr,
    utils::with_span,
    value::{Value, ValuePtr},
};

#[repr(transparent)]
#[derive(Debug, Clone, Copy, Eq, PartialEq, Hash)]
pub struct CheckVar(usize);

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

#[derive(Debug)]
pub enum MetaEntry {
    Unsolved { blocking: SetUsize, ty: ValuePtr },
    Solved { val: ValuePtr, ty: ValuePtr },
}

#[derive(Debug)]
pub struct MetaContext {
    checks: RefCell<Vec<CheckEntry>>,
    metas: RefCell<Vec<MetaEntry>>,
}

impl Default for MetaContext {
    fn default() -> Self {
        Self::new()
    }
}

impl MetaContext {
    pub fn new() -> Self {
        Self {
            checks: RefCell::new(Vec::new()),
            metas: RefCell::new(Vec::new()),
        }
    }
    pub fn new_check(&self, ctx: Context, term: TermPtr, ty: ValuePtr, meta: MetaVar) -> CheckVar {
        let mut checks = self.checks.borrow_mut();
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
        let metas = self.metas.borrow();
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
        let checks = self.checks.borrow();
        match checks.get(var.0) {
            Some(CheckEntry::Checked(term)) => env.evaluate(term.clone(), self),
            Some(CheckEntry::Unchecked { ctx, meta, .. }) => {
                env.app_pruning(self.get_meta_value(*meta, span)?, &ctx.pruning, self)
            }
            None => Err(Error::internal("invalid check variable")),
        }
    }
    pub fn get_check<F, R>(&self, var: CheckVar) -> Result<CheckEntry> {
        let checks = self.checks.borrow();
        checks
            .get(var.0)
            .cloned()
            .ok_or_else(|| Error::internal("invalid check variable"))
    }
    pub fn modify_check<F>(&self, var: CheckVar, conti: F) -> Result<()>
    where
        F: FnOnce(&mut CheckEntry) -> Result<()>,
    {
        let mut checks = self.checks.borrow_mut();
        match checks.get_mut(var.0) {
            Some(entry) => conti(entry),
            None => Err(Error::internal("invalid check variable")),
        }
    }
    pub fn new_meta(&self, ty: ValuePtr) -> MetaVar {
        let mut metas = self.metas.borrow_mut();
        let var = MetaVar(metas.len());
        metas.push(MetaEntry::Unsolved {
            blocking: SetUsize::new(),
            ty,
        });
        var
    }
    pub fn modify_meta<F>(&self, var: MetaVar, conti: F) -> Result<()>
    where
        F: FnOnce(&mut MetaEntry) -> Result<()>,
    {
        let mut metas = self.metas.borrow_mut();
        match metas.get_mut(var.0) {
            Some(entry) => conti(entry),
            None => Err(Error::internal("invalid meta variable".to_string())),
        }
    }
    pub fn add_blocker(&self, chk: CheckVar, meta: MetaVar) -> Result<()> {
        let mut metas = self.metas.borrow_mut();
        match metas.get_mut(meta.0) {
            Some(MetaEntry::Unsolved { blocking, .. }) => {
                blocking.insert(chk.0);
                Ok(())
            }
            Some(MetaEntry::Solved { .. }) => {
                Err(Error::internal("meta variable already solved".to_string()))
            }
            None => Err(Error::internal("invalid meta variable".to_string())),
        }
    }
}
