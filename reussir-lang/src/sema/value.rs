use std::rc::Rc;

use archery::RcK;
use chumsky::span::SimpleSpan;
use rpds::{HashTrieMap, Queue};
use rustc_hash::FxRandomState;

use crate::syntax::WithSpan;

use super::{
    Context, MetaEntry, UniqueName,
    eval::Environment,
    term::{Term, TermPtr},
};

pub type Map<K, V> = HashTrieMap<K, V, RcK, FxRandomState>;
pub type ValuePtr = Rc<WithSpan<Value>>;

pub type Closure = dyn for<'glb> Fn(UniqueName, ValuePtr, &'glb Context) -> ValuePtr;

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

thread_local! {
    static EMPTY_SPINE : Queue<ValuePtr> = Queue::new();
}

pub(crate) fn empty_spine() -> Queue<ValuePtr> {
    EMPTY_SPINE.with(Clone::clone)
}

impl Value {
    /// Equivalent under eta-expanding and beta-reduction
    fn is_equivalent(x: ValuePtr, y: ValuePtr, env: Environment<'_>) -> bool {
        let get_fresh_var = |n: &UniqueName| {
            let fresh = n.fresh_in(|x| env.contains_name(x));
            let var = Rc::new(WithSpan(
                Value::Rigid(fresh.clone(), empty_spine()),
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
pub(crate) mod test {
    use chumsky::span::SimpleSpan;

    use crate::sema::eval::evaluate;

    use super::*;

    pub(crate) fn lam<F, const N: usize>(name: [&str; N], body: F) -> TermPtr
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

    pub(crate) fn app<const N: usize>(f: TermPtr, x: [TermPtr; N]) -> TermPtr {
        let fake_span = SimpleSpan::new(0, 0);
        x.into_iter()
            .fold(f, |f, x| Rc::new(WithSpan(Term::App(f, x), fake_span)))
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
