use crate::{
    Result,
    meta::MetaContext,
    term::{Term, TermPtr},
    utils::{Closure, DBIdx, DBLvl, Icit, Pruning, Span, Spine, with_span, with_span_as},
    value::{Value, ValuePtr},
};
use rpds::Vector;
use tracing::trace;

#[derive(Clone, Debug)]
pub struct Environment(Vector<ValuePtr>);

impl Default for Environment {
    fn default() -> Self {
        Self::new()
    }
}

impl Environment {
    pub fn new() -> Self {
        Self(Vector::new())
    }
    pub fn with_var<F, R>(&mut self, value: ValuePtr, f: F) -> R
    where
        F: FnOnce(&mut Environment) -> R,
    {
        self.0.push_back_mut(value);
        let res = f(self);
        self.0.drop_last_mut();
        res
    }
    pub fn push_back_mut(&mut self, value: ValuePtr) {
        self.0.push_back_mut(value);
    }
    pub fn pop_back_mut(&mut self) {
        self.0.drop_last_mut();
    }
    pub fn push_back(&self, value: ValuePtr) -> Self {
        Self(self.0.push_back(value))
    }
    pub fn get_var(&self, idx: DBIdx) -> ValuePtr {
        assert!(
            idx.0 < self.0.len(),
            "Variable index {} out of bounds ({} in env)",
            idx.0,
            self.0.len()
        );
        let level = idx.to_level(self.0.len());
        self.0
            .get(level.0)
            .cloned()
            .expect("Variable not found in environment")
    }
    pub fn evaluate(&mut self, term: TermPtr, meta: &MetaContext) -> Result<ValuePtr> {
        match term.data() {
            Term::Var(idx) => Ok(self.get_var(*idx)),
            Term::Lambda(name, icit, body) => {
                let closure = Closure::new(self.clone(), body.clone());
                Ok(with_span_as(Value::Lambda(*name, *icit, closure), term))
            }
            Term::App(lhs, rhs, icit) => {
                let lhs = self.evaluate(lhs.clone(), meta)?;
                let rhs = self.evaluate(rhs.clone(), meta)?;
                app_val(lhs, rhs, *icit, meta, term.span)
            }
            Term::AppPruning(term, pruning) => {
                let span = term.span;
                let term = self.evaluate(term.clone(), meta)?;
                self.app_pruning(term, pruning, meta, span)
            }
            Term::Universe => Ok(with_span_as(Value::Universe, term)),
            Term::Pi(name, icit, ty, body) => {
                let closure = Closure::new(self.clone(), body.clone());
                let ty = self.evaluate(ty.clone(), meta)?;
                Ok(with_span_as(Value::Pi(*name, *icit, ty, closure), term))
            }
            Term::Let { term, body, .. } => {
                let term = self.evaluate(term.clone(), meta)?;
                self.with_var(term, |env| env.evaluate(body.clone(), meta))
            }
            Term::Meta(m) => Ok(meta.get_meta_value(*m, term.span)),
            Term::Postponed(c) => meta.get_check_value(self, *c, term.span),
        }
    }
    pub fn app_pruning(
        &self,
        value: ValuePtr,
        pruning: &Pruning,
        meta: &MetaContext,
        span: Span,
    ) -> Result<ValuePtr> {
        assert_eq!(self.0.len(), pruning.len(), "pruning length mismatch");
        pruning
            .iter()
            .rev()
            .copied()
            .zip(self.0.iter().rev().cloned())
            .filter_map(|(a, b)| a.map(|i| (i, b)))
            .try_fold(value, |acc, (icit, value)| {
                app_val(acc, value, icit, meta, span)
            })
    }
    pub fn normalize(&mut self, term: TermPtr, meta: &MetaContext) -> Result<TermPtr> {
        let value = self.evaluate(term.clone(), meta)?;
        quote(DBLvl(self.0.len()), value.clone(), meta)
    }
}

pub(crate) fn app_val(
    lhs: ValuePtr,
    rhs: ValuePtr,
    icit: Icit,
    meta: &MetaContext,
    span: Span,
) -> Result<ValuePtr> {
    match lhs.data() {
        Value::Lambda(_, _, closure) => closure.apply(rhs, meta),
        Value::Flex(meta, spine) => Ok(with_span(
            Value::Flex(*meta, spine.push_back((rhs, icit))),
            span,
        )),
        Value::Rigid(name, spine) => Ok(with_span(
            Value::Rigid(*name, spine.push_back((rhs, icit))),
            span,
        )),
        _ => unreachable!("lhs is not applicable"),
    }
}

pub(crate) fn app_spine(
    value: ValuePtr,
    spine: &Spine,
    meta: &MetaContext,
    span: Span,
) -> Result<ValuePtr> {
    spine.iter().cloned().try_fold(value, |acc, (arg, icit)| {
        app_val(acc, arg, icit, meta, span)
    })
}

fn quote_spine(
    level: DBLvl,
    term: TermPtr,
    spine: &Spine,
    mctx: &MetaContext,
    span: Span,
) -> Result<TermPtr> {
    spine.iter().try_fold(term, |acc, (arg, icit)| {
        let arg = quote(level, arg.clone(), mctx)?;
        Ok(with_span(Term::App(acc, arg, *icit), span))
    })
}

pub fn quote(level: DBLvl, value: ValuePtr, mctx: &MetaContext) -> Result<TermPtr> {
    match value.data() {
        Value::Flex(meta, spine) => quote_spine(
            level,
            with_span(Term::Meta(*meta), value.span),
            spine,
            mctx,
            value.span,
        ),
        Value::Rigid(var, spine) => quote_spine(
            level,
            with_span(Term::Var(var.to_index(level)), value.span),
            spine,
            mctx,
            value.span,
        ),
        Value::Lambda(name, icit, closure) => {
            let arg = with_span(Value::var(level), name.span);
            let body = closure.apply(arg, mctx)?;
            Ok(with_span_as(
                Term::Lambda(*name, *icit, quote(level.next(), body, mctx)?),
                value,
            ))
        }
        Value::Pi(name, icit, arg_ty, closure) => {
            let arg_ty = quote(level, arg_ty.clone(), mctx)?;
            let arg = with_span(Value::var(level), name.span);
            let body = closure.apply(arg, mctx)?;
            Ok(with_span_as(
                Term::Pi(*name, *icit, arg_ty, quote(level.next(), body, mctx)?),
                value,
            ))
        }
        Value::Universe => Ok(with_span_as(Term::Universe, value)),
    }
}
