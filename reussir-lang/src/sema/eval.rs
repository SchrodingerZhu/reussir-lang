use std::rc::Rc;

use archery::RcK;
use chumsky::span::SimpleSpan;
use rpds::{HashTrieMap, Queue};
use rustc_hash::FxRandomState;

use crate::syntax::WithSpan;

use super::{
    Context, MetaEntry, UniqueName,
    term::{Term, TermPtr},
    value::{Map, Value, ValuePtr, empty_spine},
};

#[derive(Clone)]
pub struct Environment<'a> {
    bindings: Map<UniqueName, ValuePtr>,
    global_ctx: &'a Context,
}

impl<'a> Environment<'a> {
    pub fn new(global_ctx: &'a Context) -> Self {
        Self {
            bindings: Map::new_with_hasher_and_ptr_kind(FxRandomState::new()),
            global_ctx,
        }
    }
    pub fn new_with(global_ctx: &'a Context, bindings: Map<UniqueName, ValuePtr>) -> Self {
        Self {
            bindings,
            global_ctx,
        }
    }
    pub fn names(&self) -> impl Iterator<Item = &UniqueName> {
        self.bindings.keys()
    }
    pub fn values(&self) -> impl Iterator<Item = &ValuePtr> {
        self.bindings.values()
    }
    pub fn select_values<'b>(
        &'a self,
        iter: impl Iterator<Item = &'b UniqueName>,
    ) -> impl Iterator<Item = &'a ValuePtr> {
        iter.map(|x| {
            self.bindings
                .get(x)
                .expect("bound value should always be defined")
        })
    }
    pub fn lookup_name(&self, name: &UniqueName) -> Option<&ValuePtr> {
        self.bindings.get(name)
    }
    pub fn contains_name(&self, name: &UniqueName) -> bool {
        self.bindings.contains_key(name)
    }
    pub fn define(&self, name: UniqueName, value: ValuePtr) -> Self {
        Self {
            bindings: self.bindings.insert(name, value),
            global_ctx: self.global_ctx,
        }
    }
    pub fn bindings(&self) -> Map<UniqueName, ValuePtr> {
        self.bindings.clone()
    }
    pub fn global(&self) -> &'a Context {
        self.global_ctx
    }
    pub fn lookup_meta(&self, idx: usize) -> ValuePtr {
        let Some(meta) = self.global_ctx.lookup_meta(idx) else {
            tracing::error!("meta {idx} is not found in global context");
            return Rc::new(WithSpan(Value::Invalid, SimpleSpan::new(0, 0)));
        };
        match meta {
            MetaEntry::Unsolved(span) => Rc::new(WithSpan(Value::Flex(idx, empty_spine()), span)),
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

pub fn evaluate(ctx: Environment<'_>, term: TermPtr) -> ValuePtr {
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

pub fn quote(value: ValuePtr, global: &Context) -> TermPtr {
    let quote = |value| {
        stacker::maybe_grow(32 * 1024, 1024 * 1024, || {
            crate::sema::eval::quote(value, global)
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
                Value::Rigid(name.clone(), empty_spine()),
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
                Value::Rigid(name.clone(), empty_spine()),
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

#[cfg(test)]
mod test {
    use chumsky::span::SimpleSpan;

    use super::*;
    use crate::sema::value::test::*;

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
}
