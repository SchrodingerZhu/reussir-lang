use std::{ops::Deref, rc::Rc};

use super::{FieldName, UniqueName, Vec};
use archery::RcK;
use chumsky::span::SimpleSpan;
use gc_arena::{Arena, Collect, Gc, Mutation, Rootable, Static, allocator_api::MetricsAlloc, lock};
use rpds::HashTrieMap;
use rustc_hash::FxRandomState;

use crate::syntax::WithSpan;

use super::{Box, QualifiedName, UStr};

pub type TermPtr<'gc> = Gc<'gc, WithSpan<Term<'gc>>>;
pub type ValuePtr<'gc> = Gc<'gc, WithSpan<Value<'gc>>>;

#[derive(Clone, Collect)]
#[collect(no_drop)]
pub enum Term<'gc> {
    /// Integer literal
    Integer(Static<rug::Integer>),
    /// Float literal
    Float(Static<rug::Float>),
    /// String literal
    Str(UStr),
    /// Boolean literal
    Boolean(bool),
    /// function call
    FuncCall {
        target: WithSpan<QualifiedName<'gc>>,
        ty_args: Box<'gc, [TermPtr<'gc>]>,
        arguments: Box<'gc, [TermPtr<'gc>]>,
    },
    /// constructor call
    CtorCall {
        target: QualifiedName<'gc>,
        ty_args: Box<'gc, [TermPtr<'gc>]>,
        arguments: Box<'gc, [(FieldName, TermPtr<'gc>)]>,
    },
    /// closure call
    ClosureCall(TermPtr<'gc>, TermPtr<'gc>),
    /// project a field out of a record
    Proj {
        value: TermPtr<'gc>,
        field: FieldName,
    },
    /// match variant
    Match {},
    /// cast variant
    Cast {},
    /// lift function into a closure
    FuncAbs {
        target: QualifiedName<'gc>,
        ty_args: Box<'gc, [TermPtr<'gc>]>,
    },
    /// lift constructor into a closure
    CtorAbs {
        target: QualifiedName<'gc>,
        ty_args: Box<'gc, [TermPtr<'gc>]>,
    },
    /// Lambda expression
    Lambda {
        binding: Gc<'gc, List<'gc, UniqueName<'gc>>>,
        body: TermPtr<'gc>,
    },
    /// Let binding
    Let {
        name: UniqueName<'gc>,
        binding: TermPtr<'gc>,
        body: TermPtr<'gc>,
    },
    /// Sequence
    Seq(TermPtr<'gc>, TermPtr<'gc>),
    /// Builtin Integer types,
    IntTy(Static<crate::syntax::r#type::Int>),
    /// Builtin Float types
    FloatTy(Static<crate::syntax::r#type::Float>),
    /// Pi type
    Pi {
        name: UniqueName<'gc>,
        arg: TermPtr<'gc>,
        body: TermPtr<'gc>,
    },
    Var(UniqueName<'gc>),
    StrTy,
    BooleanTy,
    Universe,
    MetaVar(UniqueName<'gc>),
    CheckVar,
    Invalid,
}

type Map<K, V> = HashTrieMap<K, V, RcK, FxRandomState>;

#[derive(Clone)]
pub struct Closure<'gc> {
    variables: Map<UniqueName<'gc>, ValuePtr<'gc>>,
    body: TermPtr<'gc>,
}

unsafe impl<'gc> Collect<'gc> for Closure<'gc> {
    const NEEDS_TRACE: bool = true;

    fn trace<T: gc_arena::collect::Trace<'gc>>(&self, cc: &mut T) {
        for (k, v) in self.variables.iter() {
            k.trace(cc);
            v.trace(cc);
        }
    }
}

impl std::fmt::Display for Term<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Term::Integer(_) => todo!(),
            Term::Float(_) => todo!(),
            Term::Str(_) => todo!(),
            Term::Boolean(_) => todo!(),
            Term::FuncCall {
                target,
                ty_args,
                arguments,
            } => todo!(),
            Term::CtorCall {
                target,
                ty_args,
                arguments,
            } => todo!(),
            Term::ClosureCall(gc, gc1) => {
                write!(f, "({} {})", ***gc, ***gc1)
            }
            Term::Proj { value, field } => todo!(),
            Term::Match {} => todo!(),
            Term::Cast {} => todo!(),
            Term::FuncAbs { target, ty_args } => todo!(),
            Term::CtorAbs { target, ty_args } => todo!(),
            Term::Lambda { binding, body } => {
                write!(f, "(λ")?;
                let mut binding = *binding;
                while let List::Cons(hd, tail) = &*binding {
                    write!(f, "{}", hd.0.0.0)?;
                    binding = *tail;
                    if !binding.is_empty() {
                        write!(f, " ");
                    }
                }
                write!(f, ".")?;
                body.0.fmt(f)?;
                write!(f, ")")
            }
            Term::Let {
                name,
                binding,
                body,
            } => todo!(),
            Term::Seq(gc, gc1) => todo!(),
            Term::IntTy(_) => todo!(),
            Term::FloatTy(_) => todo!(),
            Term::Pi { name, arg, body } => todo!(),
            Term::Var(unique_name) => {
                write!(f, "{}", unique_name.0.0.0)
            }
            Term::StrTy => todo!(),
            Term::BooleanTy => todo!(),
            Term::Universe => todo!(),
            Term::MetaVar(unique_name) => todo!(),
            Term::CheckVar => todo!(),
            Term::Invalid => write!(f, "<invalid>"),
        }
    }
}

#[derive(Clone, Collect)]
#[collect(no_drop)]
pub enum Value<'gc> {
    /// cannot elaborate
    Stuck(TermPtr<'gc>),
    Var(UniqueName<'gc>),
    ClosureCall(ValuePtr<'gc>, ValuePtr<'gc>),
    Universe,
    Pi {
        name: UniqueName<'gc>,
        arg: ValuePtr<'gc>,
        body: Closure<'gc>,
    },
    Lambda {
        binding: Gc<'gc, List<'gc, UniqueName<'gc>>>,
        body: Closure<'gc>,
    },
    Invalid,
}

#[derive(Clone, Collect)]
#[collect(no_drop)]
pub enum List<'gc, T> {
    Cons(T, Gc<'gc, Self>),
    Nil,
}

impl<'gc, T: Collect<'gc>> List<'gc, T> {
    pub fn nil(mc: &Mutation<'gc>) -> Gc<'gc, Self> {
        Gc::new(mc, Self::Nil)
    }
    pub fn cons(mc: &Mutation<'gc>, head: T, tail: Gc<'gc, Self>) -> Gc<'gc, Self> {
        Gc::new(mc, Self::Cons(head, tail))
    }
    pub fn is_empty(&self) -> bool {
        matches!(self, Self::Nil)
    }
    pub fn head(&self) -> Option<&T> {
        if let Self::Cons(hd, _) = self {
            Some(hd)
        } else {
            None
        }
    }
    pub fn tail(xs: Gc<'gc, Self>) -> Gc<'gc, Self> {
        if let Self::Cons(_, tail) = *xs {
            tail
        } else {
            xs
        }
    }
}

struct ListIter<'gc, T>(Gc<'gc, List<'gc, T>>);
impl<'gc, T> Iterator for ListIter<'gc, T> {
    type Item = &'gc T;

    fn next(&mut self) -> Option<Self::Item> {
        let list = self.0;
        match list.as_ref() {
            List::Cons(head, tail) => {
                self.0 = *tail;
                Some(head)
            }
            List::Nil => None,
        }
    }
}

#[derive(Clone)]
pub struct EvalContext<'gc> {
    bindings: Map<UniqueName<'gc>, ValuePtr<'gc>>,
}

impl<'gc> EvalContext<'gc> {
    fn with(&self, name: UniqueName<'gc>, value: ValuePtr<'gc>) -> Self {
        let bindings = self.bindings.insert(name, value);
        Self { bindings }
    }
}

unsafe impl<'gc> Collect<'gc> for EvalContext<'gc> {
    const NEEDS_TRACE: bool = true;

    fn trace<T: gc_arena::collect::Trace<'gc>>(&self, cc: &mut T) {
        for (k, v) in self.bindings.iter() {
            k.trace(cc);
            v.trace(cc);
        }
    }
}

fn evaluate<'gc>(mc: &Mutation<'gc>, ctx: EvalContext<'gc>, term: TermPtr<'gc>) -> ValuePtr<'gc> {
    match &**term {
        Term::Var(name) => ctx.bindings.get(name).copied().unwrap_or_else(|| {
            tracing::error!(
                "failed to resolve {name:?} in {:?}",
                ctx.bindings.keys().collect::<std::vec::Vec<_>>()
            );
            Gc::new(mc, WithSpan(Value::Invalid, term.1))
        }),
        Term::Universe => Gc::new(mc, WithSpan(Value::Universe, term.1)),
        Term::Let {
            name,
            binding,
            body,
        } => {
            let bound = evaluate(mc, ctx.clone(), *binding);
            let ctx = ctx.with(*name, bound);
            evaluate(mc, ctx, *body)
        }
        Term::Lambda { binding, body } => Gc::new(
            mc,
            WithSpan(
                Value::Lambda {
                    binding: *binding,
                    body: Closure {
                        variables: ctx.bindings,
                        body: *body,
                    },
                },
                term.1,
            ),
        ),
        Term::Pi { name, arg, body } => Gc::new(
            mc,
            WithSpan(
                Value::Pi {
                    name: *name,
                    arg: evaluate(mc, ctx.clone(), *arg),
                    body: Closure {
                        variables: ctx.bindings,
                        body: *body,
                    },
                },
                term.1,
            ),
        ),
        Term::ClosureCall(lam, arg) => {
            let lam = evaluate(mc, ctx.clone(), *lam);
            let arg = evaluate(mc, ctx.clone(), *arg);
            match &**lam {
                Value::Lambda { binding, body } => {
                    let variables = body.variables.insert(binding.head().cloned().unwrap(), arg);
                    let remaining = List::tail(*binding);

                    if remaining.is_empty() {
                        evaluate(
                            mc,
                            EvalContext {
                                bindings: variables,
                            },
                            body.body,
                        )
                    } else {
                        Gc::new(
                            mc,
                            WithSpan(
                                Value::Lambda {
                                    binding: remaining,
                                    body: Closure {
                                        variables,
                                        body: body.body,
                                    },
                                },
                                term.1,
                            ),
                        )
                    }
                }
                _ => Gc::new(mc, WithSpan(Value::ClosureCall(lam, arg), term.1)),
            }
        }
        _ => Gc::new(mc, WithSpan(Value::Stuck(term), term.1)),
    }
}

fn quote<'gc>(mc: &Mutation<'gc>, value: ValuePtr<'gc>) -> TermPtr<'gc> {
    match &**value {
        Value::Stuck(term) => *term,
        Value::Var(unique_name) => Gc::new(mc, WithSpan(Term::Var(*unique_name), value.1)),
        Value::ClosureCall(lam, arg) => {
            let lam = quote(mc, *lam);
            let arg = quote(mc, *arg);
            Gc::new(mc, WithSpan(Term::ClosureCall(lam, arg), value.1))
        }
        Value::Universe => Gc::new(mc, WithSpan(Term::Universe, value.1)),
        Value::Pi { name, arg, body } => {
            let arg = quote(mc, *arg);
            let var = Gc::new(mc, WithSpan(Value::Var(*name), arg.1));
            let env = EvalContext {
                bindings: body.variables.clone(),
            }
            .with(*name, var);
            let body = quote(mc, evaluate(mc, env, body.body));
            Gc::new(
                mc,
                WithSpan(
                    Term::Pi {
                        name: *name,
                        arg,
                        body,
                    },
                    value.1,
                ),
            )
        }
        Value::Lambda { binding, body } => {
            let env = EvalContext {
                bindings: body.variables.clone(),
            };
            let span = value.1;
            let env = ListIter(*binding).fold(env, |acc: EvalContext, var| {
                let value = Gc::new(mc, WithSpan(Value::Var(*var), span));
                acc.with(*var, value)
            });
            let body = quote(mc, evaluate(mc, env, body.body));
            Gc::new(
                mc,
                WithSpan(
                    Term::Lambda {
                        binding: *binding,
                        body,
                    },
                    value.1,
                ),
            )
        }
        Value::Invalid => Gc::new(mc, WithSpan(Term::Invalid, value.1)),
    }
}

#[cfg(test)]
mod test {
    use super::*;

    fn lam<'a, F>(mc: &Mutation<'a>, name: &[&str], body: F) -> TermPtr<'a>
    where
        F: FnOnce(&Mutation<'a>, &[TermPtr<'a>]) -> TermPtr<'a>,
    {
        let fake_span = SimpleSpan::new(0, 0);
        let names = name
            .iter()
            .copied()
            .map(|name| UniqueName::new(mc, name, fake_span))
            .collect::<std::vec::Vec<_>>();

        let vars = names
            .iter()
            .map(|name| Gc::new(mc, WithSpan(Term::Var(*name), fake_span)))
            .collect::<std::vec::Vec<_>>();

        let binding = names
            .iter()
            .copied()
            .rev()
            .fold(List::nil(mc), |acc, x| List::cons(mc, x, acc));

        Gc::new(
            mc,
            WithSpan(
                Term::Lambda {
                    binding,
                    body: body(mc, &vars),
                },
                fake_span,
            ),
        )
    }

    fn app<'a>(mc: &Mutation<'a>, f: TermPtr<'a>, x: TermPtr<'a>) -> TermPtr<'a> {
        let fake_span = SimpleSpan::new(0, 0);
        Gc::new(mc, WithSpan(Term::ClosureCall(f, x), fake_span))
    }

    #[test]
    fn it_normalizes_app() {
        tracing_subscriber::fmt::init();
        let mut arena = Arena::<Rootable![TermPtr<'_>]>::new(|mc| {
            let identity = lam(mc, &["x"], |_, x| x[0]);
            let fx = lam(mc, &["f", "x"], |mc, args| app(mc, args[0], args[1]));
            let res = app(mc, fx, identity);
            assert_eq!(res.to_string(), "((λf x.(f x)) (λx.x))");
            res
        });
        arena.finish_cycle();
        arena.mutate_root(|mc, root| {
            let context = EvalContext {
                bindings: Map::new_with_hasher_and_ptr_kind(FxRandomState::new()),
            };
            *root = quote(mc, evaluate(mc, context, *root));
            assert_eq!(root.to_string(), "(λx.x)");
        });
        arena.finish_cycle();
    }
}
