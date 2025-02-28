use std::rc::Rc;

use archery::RcK;
use rpds::HashTrieMap;
use rustc_hash::FxRandomState;

use crate::syntax::WithSpan;

use super::{
    UniqueName,
    term::{Term, TermPtr},
};

type Map<K, V> = HashTrieMap<K, V, RcK, FxRandomState>;
pub type ValuePtr = Rc<WithSpan<Value>>;

#[derive(Clone)]
pub enum Value {
    /// cannot elaborate
    Stuck(TermPtr),
    Var(UniqueName),
    App(ValuePtr, ValuePtr),
    Universe,
    Pi {
        name: UniqueName,
        arg: ValuePtr,
        body: Rc<dyn Fn(UniqueName, ValuePtr) -> ValuePtr>,
    },
    Lambda {
        name: UniqueName,
        body: Rc<dyn Fn(UniqueName, ValuePtr) -> ValuePtr>,
    },
    Invalid,
}

type EvalContext = Map<UniqueName, ValuePtr>;

fn evaluate(ctx: EvalContext, term: TermPtr) -> ValuePtr {
    match &**term {
        Term::Var(name) => ctx.get(name).cloned().unwrap_or_else(|| {
            tracing::error!(
                "failed to resolve {name:?} in {:?}",
                ctx.keys().collect::<std::vec::Vec<_>>()
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
            let ctx = ctx.insert(name.clone(), bound);
            evaluate(ctx, body.clone())
        }
        Term::Lambda { binding, body } => {
            let ctx = ctx.clone();
            let body = body.clone();
            Rc::new(WithSpan(
                Value::Lambda {
                    name: binding.clone(),
                    body: Rc::new(move |name, arg| evaluate(ctx.insert(name, arg), body.clone())),
                },
                term.1,
            ))
        }
        Term::Pi { name, arg, body } => {
            let body = body.clone();
            let ctx = ctx.clone();
            Rc::new(WithSpan(
                Value::Pi {
                    name: name.clone(),
                    arg: evaluate(ctx.clone(), arg.clone()),
                    body: Rc::new(move |name, arg| evaluate(ctx.insert(name, arg), body.clone())),
                },
                term.1,
            ))
        }
        Term::App(lam, arg) => {
            let lam = evaluate(ctx.clone(), lam.clone());
            let arg = evaluate(ctx.clone(), arg.clone());
            match &**lam {
                Value::Lambda { name, body } => body(name.clone(), arg),
                _ => Rc::new(WithSpan(Value::App(lam, arg), term.1)),
            }
        }
        _ => Rc::new(WithSpan(Value::Stuck(term.clone()), term.1)),
    }
}

fn quote(value: ValuePtr) -> TermPtr {
    let quote =
        |value| stacker::maybe_grow(32 * 1024, 1024 * 1024, || crate::sema::norm::quote(value));
    match &**value {
        Value::Stuck(term) => term.clone(),
        Value::Var(unique_name) => Rc::new(WithSpan(Term::Var(unique_name.clone()), value.1)),
        Value::App(lam, arg) => {
            let lam = quote(lam.clone());
            let arg = quote(arg.clone());
            Rc::new(WithSpan(Term::App(lam, arg), value.1))
        }
        Value::Universe => Rc::new(WithSpan(Term::Universe, value.1)),
        Value::Pi { name, arg, body } => {
            let arg = quote(arg.clone());
            let var = Rc::new(WithSpan(Value::Var(name.clone()), name.span()));
            let body = quote(body(name.clone(), var));
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
            let var = Rc::new(WithSpan(Value::Var(name.clone()), name.span()));
            let body = quote(body(name.clone(), var));
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
    fn is_equivalent(x: ValuePtr, y: ValuePtr, env: EvalContext) -> bool {
        let get_fresh_var = |n: &UniqueName| {
            let fresh = n.fresh_in(|x| env.contains_key(x));
            let var = Rc::new(WithSpan(Value::Var(fresh.clone()), fresh.span()));
            let env = env.insert(fresh, var.clone());
            (var, env)
        };
        let check_lambda_like =
            |n0: &UniqueName,
             n1: &UniqueName,
             b0: &dyn Fn(UniqueName, ValuePtr) -> ValuePtr,
             b1: &dyn Fn(UniqueName, ValuePtr) -> ValuePtr| {
                let (var, env) = get_fresh_var(n0);
                let b0 = b0(n0.clone(), var.clone());
                let b1 = b1(n1.clone(), var);
                Value::is_equivalent(b0, b1, env)
            };
        let app = |f: ValuePtr, x: ValuePtr| {
            let span = f.1;
            Rc::new(WithSpan(Value::App(f, x), span))
        };
        match (&x.0, &y.0) {
            (Value::Stuck(term0), Value::Stuck(term1)) => term0.is_alpha_equivalent(term1),
            (Value::Var(name0), Value::Var(name1)) => name0 == name1,
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
                let mut s = b(n.clone(), var.clone());
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
        let context = Map::new_with_hasher_and_ptr_kind(FxRandomState::new());

        let res = quote(evaluate(context, res));
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
        let context = Map::new_with_hasher_and_ptr_kind(FxRandomState::new());

        let res = quote(evaluate(context, thousand));
        let buf = res.to_string();
        println!("{}", buf);
        assert_eq!(buf.chars().filter(|x| *x == 's').count(), 1001);
    }

    #[test]
    fn it_checks_simple_equivalence() {
        let context = Map::new_with_hasher_and_ptr_kind(FxRandomState::new());
        let idx = evaluate(context.clone(), lam(["x"], |[x]| x));
        let idy = evaluate(context.clone(), lam(["y"], |[y]| y));
        assert!(Value::is_equivalent(idx, idy, context))
    }

    #[test]
    fn it_checks_equivalence_after_eta() {
        let context = Map::new_with_hasher_and_ptr_kind(FxRandomState::new());
        let id = lam(["x"], |[x]| x);
        let id_plain = evaluate(context.clone(), id.clone());
        let id_eta = evaluate(context.clone(), lam(["x"], move |[x]| app(id, [x])));
        assert!(Value::is_equivalent(id_plain, id_eta, context))
    }
}
