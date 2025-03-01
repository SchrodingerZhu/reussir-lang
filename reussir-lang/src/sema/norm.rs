use std::rc::Rc;

use archery::RcK;
use chumsky::span::SimpleSpan;
use rpds::{HashTrieMap, Queue, Stack, Vector};
use rustc_hash::FxRandomState;

use crate::syntax::WithSpan;

use super::{
    Context, MetaEntry, UniqueName,
    term::{Term, TermPtr},
};

type Map<K, V> = HashTrieMap<K, V, RcK, FxRandomState>;
pub type ValuePtr = Rc<WithSpan<Value>>;

type Closure = dyn for<'glb> Fn(UniqueName, ValuePtr, &'glb Context) -> ValuePtr;

#[derive(Clone)]
pub enum Value {
    /// Stuck due to effectful terms
    Stuck(TermPtr),
    /// Stuck on variable
    Rigid(UniqueName, Queue<ValuePtr>),
    /// Stuck on unsolved meta
    Flex(usize, Queue<ValuePtr>),
    App(ValuePtr, ValuePtr),
    Universe,
    Pi {
        name: UniqueName,
        arg: ValuePtr,
        body: Rc<Closure>,
    },
    Lambda {
        name: UniqueName,
        body: Rc<Closure>,
    },
    Invalid,
}

#[derive(Clone)]
struct Environment<'a> {
    bindings: Map<UniqueName, ValuePtr>,
    global_ctx: &'a Context,
}

thread_local! {
    static EMPTY_VALUE_QUEUE : Queue<ValuePtr> = Queue::new();
}

impl<'a> Environment<'a> {
    fn new(global_ctx: &'a Context) -> Self {
        Self {
            bindings: Map::new_with_hasher_and_ptr_kind(FxRandomState::new()),
            global_ctx,
        }
    }
    fn new_with(global_ctx: &'a Context, bindings: Map<UniqueName, ValuePtr>) -> Self {
        Self {
            bindings,
            global_ctx,
        }
    }
    fn names(&self) -> impl Iterator<Item = &UniqueName> {
        self.bindings.keys()
    }
    fn values(&self) -> impl Iterator<Item = &ValuePtr> {
        self.bindings.values()
    }
    fn select_values<'b>(
        &'a self,
        iter: impl Iterator<Item = &'b UniqueName>,
    ) -> impl Iterator<Item = &'a ValuePtr> {
        iter.map(|x| {
            self.bindings
                .get(x)
                .expect("bound value should always be defined")
        })
    }
    fn lookup_name(&self, name: &UniqueName) -> Option<&ValuePtr> {
        self.bindings.get(name)
    }
    fn contains_name(&self, name: &UniqueName) -> bool {
        self.bindings.contains_key(name)
    }
    fn define(&self, name: UniqueName, value: ValuePtr) -> Self {
        Self {
            bindings: self.bindings.insert(name, value),
            global_ctx: self.global_ctx,
        }
    }
    fn bindings(&self) -> Map<UniqueName, ValuePtr> {
        self.bindings.clone()
    }
    fn global(&self) -> &'a Context {
        self.global_ctx
    }
    fn lookup_meta(&self, idx: usize) -> ValuePtr {
        let Some(meta) = self.global_ctx.lookup_meta(idx) else {
            tracing::error!("meta {idx} is not found in global context");
            return Rc::new(WithSpan(Value::Invalid, SimpleSpan::new(0, 0)));
        };
        match meta {
            MetaEntry::Unsolved(span) => Rc::new(WithSpan(
                Value::Flex(idx, EMPTY_VALUE_QUEUE.with(Clone::clone)),
                span,
            )),
            MetaEntry::Solved(x) => x,
        }
    }
}

fn value_apply(f: ValuePtr, x: ValuePtr, span: SimpleSpan, global: &Context) -> ValuePtr {
    match &**f {
        Value::Lambda { name, body } => body(name.clone(), x.clone(), global),
        Value::Flex(meta, spine) => {
            Rc::new(WithSpan(Value::Flex(*meta, spine.enqueue(x.clone())), span))
        }
        Value::Rigid(var, spine) => Rc::new(WithSpan(
            Value::Rigid(var.clone(), spine.enqueue(x.clone())),
            span,
        )),
        _ => {
            tracing::error!("attempt to apply non-applicable values {span}");
            Rc::new(WithSpan(Value::Invalid, span))
        }
    }
}

fn value_apply_multiple<I>(f: ValuePtr, iter: I, span: SimpleSpan, global: &Context) -> ValuePtr
where
    I: Iterator<Item = ValuePtr>,
{
    iter.fold(f, |acc, x| value_apply(acc, x, span, global))
}

fn value_apply_selectively<'a>(
    f: ValuePtr,
    iter: impl Iterator<Item = &'a UniqueName>,
    span: SimpleSpan,
    ctx: &Environment,
) -> ValuePtr {
    value_apply_multiple(f, ctx.select_values(iter).cloned(), span, ctx.global())
}

fn evaluate(ctx: Environment<'_>, term: TermPtr) -> ValuePtr {
    match &**term {
        Term::Var(name) => ctx.lookup_name(name).cloned().unwrap_or_else(|| {
            tracing::error!(
                "failed to resolve {name:?} in {:?}",
                ctx.names().collect::<std::vec::Vec<_>>()
            );
            Rc::new(WithSpan(Value::Invalid, term.1))
        }),
        Term::Universe => Rc::new(WithSpan(Value::Universe, term.1)),
        Term::Let {
            name,
            binding,
            body,
        } => {
            let bound = evaluate(ctx.clone(), binding.clone());
            let ctx = ctx.define(name.clone(), bound);
            evaluate(ctx, body.clone())
        }
        Term::Lambda { binding, body } => {
            let body = body.clone();
            let bindings = ctx.bindings();
            Rc::new(WithSpan(
                Value::Lambda {
                    name: binding.clone(),
                    body: Rc::new(move |name, arg, ctx| {
                        evaluate(
                            Environment::new_with(ctx, bindings.clone()).define(name, arg),
                            body.clone(),
                        )
                    }),
                },
                term.1,
            ))
        }
        Term::Pi { name, arg, body } => {
            let body = body.clone();
            let bindings = ctx.bindings();
            Rc::new(WithSpan(
                Value::Pi {
                    name: name.clone(),
                    arg: evaluate(ctx.clone(), arg.clone()),
                    body: Rc::new(move |name, arg, ctx| {
                        evaluate(
                            Environment::new_with(ctx, bindings.clone()).define(name, arg),
                            body.clone(),
                        )
                    }),
                },
                term.1,
            ))
        }
        Term::App(lam, arg) => {
            let lam = evaluate(ctx.clone(), lam.clone());
            let arg = evaluate(ctx.clone(), arg.clone());
            match &**lam {
                Value::Lambda { name, body } => body(name.clone(), arg, ctx.global()),
                _ => Rc::new(WithSpan(Value::App(lam, arg), term.1)),
            }
        }
        Term::Meta(x) => ctx.lookup_meta(*x),
        Term::InsertedMeta(x, bounds) => {
            let meta = ctx.lookup_meta(*x);
            value_apply_selectively(meta, bounds.iter(), term.1, &ctx)
        }
        _ => Rc::new(WithSpan(Value::Stuck(term.clone()), term.1)),
    }
}

fn force(value: ValuePtr, global: &Context) -> ValuePtr {
    match &**value {
        Value::Flex(meta, spine) => {
            if let Some(MetaEntry::Solved(solved)) = global.lookup_meta(*meta) {
                force(
                    value_apply_multiple(solved, spine.iter().cloned(), value.1, global),
                    global,
                )
            } else {
                value
            }
        }
        _ => value,
    }
}

fn quote(value: ValuePtr, global: &Context) -> TermPtr {
    let quote = |value| {
        stacker::maybe_grow(32 * 1024, 1024 * 1024, || {
            crate::sema::norm::quote(value, global)
        })
    };
    let quote_spine = |term: TermPtr, spine: &Queue<ValuePtr>| {
        spine.iter().cloned().map(quote).fold(term, |acc, x| {
            let span = acc.1;
            Rc::new(WithSpan(Term::App(acc, x), span))
        })
    };
    let value = force(value, global);
    match &**value {
        Value::Stuck(term) => term.clone(),
        Value::Rigid(name, spine) => {
            quote_spine(Rc::new(WithSpan(Term::Var(name.clone()), value.1)), spine)
        }
        Value::Flex(idx, spine) => quote_spine(Rc::new(WithSpan(Term::Meta(*idx), value.1)), spine),
        Value::App(lam, arg) => {
            let lam = quote(lam.clone());
            let arg = quote(arg.clone());
            Rc::new(WithSpan(Term::App(lam, arg), value.1))
        }
        Value::Universe => Rc::new(WithSpan(Term::Universe, value.1)),
        Value::Pi { name, arg, body } => {
            let arg = quote(arg.clone());
            let var = Rc::new(WithSpan(
                Value::Rigid(name.clone(), EMPTY_VALUE_QUEUE.with(Clone::clone)),
                name.span(),
            ));
            let body = quote(body(name.clone(), var, global));
            Rc::new(WithSpan(
                Term::Pi {
                    name: name.clone(),
                    arg,
                    body,
                },
                value.1,
            ))
        }
        Value::Lambda { name, body } => {
            let var = Rc::new(WithSpan(
                Value::Rigid(name.clone(), EMPTY_VALUE_QUEUE.with(Clone::clone)),
                name.span(),
            ));
            let body = quote(body(name.clone(), var, global));
            Rc::new(WithSpan(
                Term::Lambda {
                    binding: name.clone(),
                    body,
                },
                value.1,
            ))
        }
        Value::Invalid => Rc::new(WithSpan(Term::Invalid, value.1)),
    }
}

impl Value {
    /// Equivalent under eta-expanding and beta-reduction
    fn is_equivalent(x: ValuePtr, y: ValuePtr, env: Environment<'_>) -> bool {
        let get_fresh_var = |n: &UniqueName| {
            let fresh = n.fresh_in(|x| env.contains_name(x));
            let var = Rc::new(WithSpan(
                Value::Rigid(fresh.clone(), EMPTY_VALUE_QUEUE.with(Clone::clone)),
                fresh.span(),
            ));
            let env = env.define(fresh, var.clone());
            (var, env)
        };
        let check_lambda_like = |n0: &UniqueName, n1: &UniqueName, b0: &Closure, b1: &Closure| {
            let (var, env) = get_fresh_var(n0);
            let b0 = b0(n0.clone(), var.clone(), env.global());
            let b1 = b1(n1.clone(), var, env.global());
            Value::is_equivalent(b0, b1, env)
        };
        let app = |f: ValuePtr, x: ValuePtr| {
            let span = f.1;
            Rc::new(WithSpan(Value::App(f, x), span))
        };
        match (&x.0, &y.0) {
            (Value::Stuck(term0), Value::Stuck(term1)) => term0.is_alpha_equivalent(term1),
            (Value::Rigid(name0, _), Value::Rigid(name1, _)) => name0 == name1,
            (Value::App(f, x), Value::App(g, y)) => {
                Value::is_equivalent(f.clone(), g.clone(), env.clone())
                    && Value::is_equivalent(x.clone(), y.clone(), env)
            }
            (Value::Universe, Value::Universe) => true,
            (
                Value::Pi {
                    name: n0,
                    arg: a0,
                    body: b0,
                },
                Value::Pi {
                    name: n1,
                    arg: a1,
                    body: b1,
                },
            ) => {
                Value::is_equivalent(a0.clone(), a1.clone(), env.clone())
                    && check_lambda_like(n0, n1, &**b0, &**b1)
            }

            (Value::Lambda { name: n0, body: b0 }, Value::Lambda { name: n1, body: b1 }) => {
                check_lambda_like(n0, n1, &**b0, &**b1)
            }

            (Value::Lambda { name: n, body: b }, value)
            | (value, Value::Lambda { name: n, body: b }) => {
                let (var, env) = get_fresh_var(n);
                let mut s = b(n.clone(), var.clone(), env.global());
                let mut t = app(y.clone(), var);
                if matches!(y.0, Value::Lambda { .. }) {
                    std::mem::swap(&mut s, &mut t);
                }
                Value::is_equivalent(s, t, env)
            }

            _ => false,
        }
    }
}

#[cfg(test)]
mod test {
    use chumsky::span::SimpleSpan;

    use super::*;

    fn lam<F, const N: usize>(name: [&str; N], body: F) -> TermPtr
    where
        F: FnOnce([TermPtr; N]) -> TermPtr,
    {
        let fake_span = SimpleSpan::new(0, 0);
        let names = name.map(|name| UniqueName::new(name, fake_span));
        let vars = names
            .clone()
            .map(|name| Rc::new(WithSpan(Term::Var(name), fake_span)));
        let body = body(vars);
        names.into_iter().rev().fold(body, |body, binding| {
            Rc::new(WithSpan(Term::Lambda { binding, body }, fake_span))
        })
    }

    fn app<const N: usize>(f: TermPtr, x: [TermPtr; N]) -> TermPtr {
        let fake_span = SimpleSpan::new(0, 0);
        x.into_iter()
            .fold(f, |f, x| Rc::new(WithSpan(Term::App(f, x), fake_span)))
    }

    #[test]
    fn it_normalizes_app() {
        _ = tracing_subscriber::fmt::try_init();
        let identity = lam(["x"], |[x]| x);
        let fx = lam(["f", "x"], |[f, x]| app(f, [x]));
        let res = app(fx, [identity]);
        assert_eq!(res.to_string(), "(λf.λx.(f x) λx.x)");
        let global = Context::new();
        let env = Environment::new(&global);

        let res = quote(evaluate(env, res), &global);
        assert_eq!(res.to_string(), "λx.x");
    }

    #[test]
    fn it_normalizes_thousand() {
        _ = tracing_subscriber::fmt::try_init();
        let five = lam(["s", "z"], |[s, z]| {
            let one = app(s.clone(), [z]);
            let two = app(s.clone(), [one]);
            let three = app(s.clone(), [two]);
            let four = app(s.clone(), [three]);
            app(s, [four])
        });
        let add = lam(["a", "b", "s", "z"], |[a, b, s, z]| {
            let bsz = app(b, [s.clone(), z]);
            let r#as = app(a, [s]);
            app(r#as, [bsz])
        });
        let mul = lam(["a", "b", "s", "z"], |[a, b, s, z]| {
            let bs = app(b, [s]);
            app(a, [bs, z])
        });
        let ten = app(add, [five.clone(), five]);
        let hundred = app(mul.clone(), [ten.clone(), ten.clone()]);
        let thousand = app(mul, [ten.clone(), hundred]);
        let global = Context::new();
        let env = Environment::new(&global);

        let res = quote(evaluate(env, thousand), &global);
        let buf = res.to_string();
        println!("{}", buf);
        assert_eq!(buf.chars().filter(|x| *x == 's').count(), 1001);
    }

    #[test]
    fn it_checks_simple_equivalence() {
        let global = Context::new();
        let env = Environment::new(&global);
        let idx = evaluate(env.clone(), lam(["x"], |[x]| x));
        let idy = evaluate(env.clone(), lam(["y"], |[y]| y));
        assert!(Value::is_equivalent(idx, idy, env))
    }

    #[test]
    fn it_checks_equivalence_after_eta() {
        let global = Context::new();
        let env = Environment::new(&global);
        let id = lam(["x"], |[x]| x);
        let id_plain = evaluate(env.clone(), id.clone());
        let id_eta = evaluate(env.clone(), lam(["x"], move |[x]| app(id, [x])));
        assert!(Value::is_equivalent(id_plain, id_eta, env))
    }
}
