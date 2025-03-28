use archery::RcK;
use rpds::HashTrieSet;
use rustc_hash::FxRandomState;

use crate::surf::SurfPtr;
use crate::{
    Result,
    ctx::Context,
    eval::{Environment, app_spine},
    term::TermPtr,
    utils::{Span, with_span},
    value::{Value, ValuePtr},
};

#[repr(transparent)]
#[derive(Debug, Clone, Copy, Eq, PartialEq, Hash)]
pub struct CheckVar(pub(crate) usize);

#[repr(transparent)]
#[derive(Debug, Clone, Copy, Eq, PartialEq, Hash)]
pub struct MetaVar(pub(crate) usize);

#[derive(Debug, Clone)]
pub enum CheckEntry {
    Checked(TermPtr),
    Unchecked {
        ctx: Context,
        term: SurfPtr,
        ty: ValuePtr,
        meta: MetaVar,
    },
}

pub type BlockerSet = HashTrieSet<CheckVar, RcK, FxRandomState>;

#[derive(Clone)]
pub enum MetaEntry {
    Unsolved { blocking: BlockerSet, ty: ValuePtr },
    Solved { val: ValuePtr, ty: ValuePtr },
}

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
        term: SurfPtr,
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
    pub fn check_vars(&self) -> impl Iterator<Item = CheckVar> + use<> {
        (0..self.checks.len()).map(CheckVar)
    }
    pub fn num_checks(&self) -> usize {
        self.checks.len()
    }
    pub fn get_meta_value(&self, var: MetaVar, span: Span) -> ValuePtr {
        let metas = &self.metas;
        match metas.get(var.0) {
            Some(MetaEntry::Solved { val, .. }) => val.clone(),
            Some(MetaEntry::Unsolved { .. }) => with_span(Value::meta(var), span),
            None => unreachable!("invalid meta variable"),
        }
    }
    pub fn get_check_value(
        &self,
        env: &mut Environment,
        var: CheckVar,
        span: Span,
    ) -> Result<ValuePtr> {
        let checks = &self.checks;
        match checks.get(var.0) {
            Some(CheckEntry::Checked(term)) => env.evaluate(term.clone(), self),
            Some(CheckEntry::Unchecked { ctx, meta, .. }) => {
                env.app_pruning(self.get_meta_value(*meta, span), &ctx.pruning, self, span)
            }
            None => unreachable!("invalid check variable"),
        }
    }
    pub fn get_check(&self, var: CheckVar) -> &CheckEntry {
        let checks = &self.checks;
        checks.get(var.0).expect("invalid check variable")
    }
    pub fn set_check(&mut self, var: CheckVar, new_entry: CheckEntry) {
        let checks = &mut self.checks;
        *checks.get_mut(var.0).expect("invalid check variable") = new_entry;
    }
    pub fn new_meta(&mut self, ty: ValuePtr, blocking: BlockerSet) -> MetaVar {
        let metas = &mut self.metas;
        let var = MetaVar(metas.len());
        metas.push(MetaEntry::Unsolved { blocking, ty });
        var
    }
    pub fn set_meta(&mut self, var: MetaVar, new_entry: MetaEntry) {
        let metas = &mut self.metas;
        *metas.get_mut(var.0).expect("invalid meta variable") = new_entry;
    }
    pub fn get_meta(&self, var: MetaVar) -> &MetaEntry {
        let metas = &self.metas;
        metas.get(var.0).expect("invalid meta variable")
    }
    pub fn add_blocker(&mut self, chk: CheckVar, meta: MetaVar) {
        let metas = &mut self.metas;
        match metas.get_mut(meta.0) {
            Some(MetaEntry::Unsolved { blocking, .. }) => {
                blocking.insert_mut(chk);
            }
            Some(MetaEntry::Solved { .. }) => unreachable!("solved meta variable"),
            None => unreachable!("invalid meta variable"),
        }
    }
    pub fn force(&self, mut target: ValuePtr) -> Result<ValuePtr> {
        loop {
            match target.data() {
                Value::Flex(m, sp) => match self.metas.get(m.0) {
                    Some(MetaEntry::Unsolved { .. }) => {
                        return Ok(target);
                    }
                    Some(MetaEntry::Solved { val, .. }) => {
                        target = app_spine(val.clone(), sp, self, target.span)?;
                        continue;
                    }
                    None => unreachable!("invalid meta variable"),
                },
                _ => return Ok(target),
            }
        }
    }
}
