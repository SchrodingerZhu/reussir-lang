use std::cell::RefCell;

use tinyset::SetUsize;

use crate::{Error, Result, ctx::Context, term::TermPtr, value::ValuePtr};

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
    pub fn get_check<F, R>(&self, var: CheckVar) -> Result<CheckEntry> {
        let checks = self.checks.borrow();
        checks
            .get(var.0)
            .cloned()
            .ok_or_else(|| Error::internal("invalid check variable".to_string()))
    }
    pub fn modify_check<F>(&self, var: CheckVar, conti: F) -> Result<()>
    where
        F: FnOnce(&mut CheckEntry) -> Result<()>,
    {
        let mut checks = self.checks.borrow_mut();
        match checks.get_mut(var.0) {
            Some(entry) => conti(entry),
            None => Err(Error::internal("invalid check variable".to_string())),
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
