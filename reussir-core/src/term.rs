use crate::ctx::Context;
use crate::eval::Environment;
use crate::utils::{DBLvl, deep_recursive};
use crate::{
    meta::{CheckVar, MetaVar},
    utils::{DBIdx, Icit, Name, Pruning, WithSpan},
};
use std::collections::HashMap;
use std::fmt::{Display, Write};
use std::rc::Rc;
use tracing::trace;

pub type TermPtr = Rc<WithSpan<Term>>;

#[derive(Debug, Clone)]
pub enum Term {
    Var(DBIdx),
    Lambda(Name, Icit, TermPtr),
    App(TermPtr, TermPtr, Icit),
    AppPruning(TermPtr, Pruning),
    Universe,
    Pi(Name, Icit, TermPtr, TermPtr),
    Let {
        name: Name,
        ty: TermPtr,
        term: TermPtr,
        body: TermPtr,
    },
    Meta(MetaVar),
    Postponed(CheckVar),
}

impl Term {
    pub(crate) fn with_ctx(&self, ctx: &Context) -> TermWithCtx {
        TermWithCtx {
            term: self,
            ctx: ctx.clone(),
        }
    }
}

struct TermFmtContext<'a, 'b> {
    names: Vec<Name>,
    fmt: &'a mut std::fmt::Formatter<'b>,
}

pub(crate) struct TermWithCtx<'a> {
    term: &'a Term,
    ctx: Context,
}

impl Display for TermWithCtx<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let mut ctx = TermFmtContext::new(f);
        for (name, _) in self.ctx.locals.iter() {
            ctx.add(name.clone());
        }
        ctx.fmt_term(self.term)
    }
}

impl<'a, 'b> TermFmtContext<'a, 'b> {
    fn new(fmt: &'a mut std::fmt::Formatter<'b>) -> Self {
        Self { names: vec![], fmt }
    }

    fn add(&mut self, name: Name) {
        self.names.push(name);
    }

    fn pop(&mut self) {
        self.names.pop();
    }

    fn fmt_term(&mut self, term: &Term) -> std::fmt::Result {
        deep_recursive(|| match term {
            Term::Var(idx) => {
                if idx.0 >= self.names.len() {
                    return write!(self.fmt, "!{}", idx.0);
                }
                let rev_idx = self.names.len() - idx.0 - 1;
                match self.names.get(rev_idx) {
                    Some(name) => write!(self.fmt, "{}", name.data()),
                    None => unreachable!(),
                }
            }
            Term::Lambda(name, icit, body) => {
                if icit == &Icit::Impl {
                    write!(self.fmt, "\\{{{}}}.", name.data())?;
                } else {
                    write!(self.fmt, "\\{}.", name.data())?;
                }
                self.add(name.clone());
                self.fmt_term(body)?;
                self.pop();
                Ok(())
            }
            Term::App(lhs, rhs, icit) => {
                if let Icit::Impl = *icit {
                    self.fmt.write_char('{')?;
                    self.fmt_term(lhs)?;
                    self.fmt.write_char(' ')?;
                    self.fmt_term(rhs)?;
                    self.fmt.write_char('}')?;
                } else {
                    self.fmt.write_char('(')?;
                    self.fmt_term(lhs)?;
                    self.fmt.write_char(' ')?;
                    self.fmt_term(rhs)?;
                    self.fmt.write_char(')')?;
                }
                Ok(())
            }
            Term::AppPruning(lhs, pruning) => {
                self.fmt.write_char('[')?;
                self.fmt_term(lhs)?;
                if !pruning.is_empty() {
                    let mut index = DBIdx(pruning.len() - 1);
                    for p in pruning.iter() {
                        self.fmt.write_char(' ')?;
                        if let Some(icit) = p {
                            if Icit::Impl == *icit {
                                self.fmt.write_char('{')?;
                                self.fmt_term(&Term::Var(index))?;
                                self.fmt.write_char('}')?;
                            } else {
                                self.fmt_term(&Term::Var(index))?;
                            }
                        }
                        index = index.prev();
                    }
                }
                self.fmt.write_char(']')
            }
            Term::Universe => self.fmt.write_char('𝓤'),
            Term::Pi(name, icit, ty, body) => {
                if !name.is_empty() {
                    if icit == &Icit::Impl {
                        write!(self.fmt, "Π{{{}", name.data())?;
                    } else {
                        write!(self.fmt, "Π({}", name.data())?;
                    }
                    write!(self.fmt, " : ")?;
                }
                self.fmt_term(ty)?;
                if !name.is_empty() {
                    if *icit == Icit::Impl {
                        self.fmt.write_char('}')?;
                    } else {
                        self.fmt.write_char(')')?;
                    }
                    self.fmt.write_char('.')?;
                } else {
                    self.fmt.write_str(" -> ")?;
                }
                self.add(name.clone());
                self.fmt_term(body)?;
                self.pop();
                Ok(())
            }
            Term::Let {
                name,
                ty,
                term,
                body,
            } => {
                write!(self.fmt, "let {} : ", name.data())?;
                self.fmt_term(ty)?;
                self.fmt.write_char('=')?;
                self.fmt_term(term)?;
                self.fmt.write_char('.')?;
                self.add(name.clone());
                self.fmt_term(body)?;
                self.pop();
                Ok(())
            }
            Term::Meta(m) => {
                write!(self.fmt, "?{}", m.0)
            }
            Term::Postponed(c) => {
                write!(self.fmt, "「{}」", c.0)
            }
        })
    }
}

impl std::fmt::Display for Term {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let mut ctx = TermFmtContext::new(f);
        ctx.fmt_term(self)
    }
}
