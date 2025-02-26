use std::{ops::Deref, rc::Rc};

use super::{FieldName, QualifiedName, UniqueName};
use archery::RcK;
use chumsky::span::SimpleSpan;
use rpds::{HashTrieMap, List};
use rustc_hash::FxRandomState;
use ustr::Ustr;

use crate::syntax::WithSpan;
pub type TermPtr = Rc<WithSpan<Term>>;
pub type ValuePtr = Rc<WithSpan<Value>>;

#[derive(Clone)]
pub enum Term {
    /// Integer literal
    Integer(rug::Integer),
    /// Float literal
    Float(rug::Float),
    /// String literal
    Str(Ustr),
    /// Boolean literal
    Boolean(bool),
    /// function call
    FuncCall {
        target: WithSpan<QualifiedName>,
        ty_args: Box<[TermPtr]>,
        arguments: Box<[TermPtr]>,
    },
    /// constructor call
    CtorCall {
        target: QualifiedName,
        ty_args: Box<[TermPtr]>,
        arguments: Box<[(FieldName, TermPtr)]>,
    },
    /// closure call
    App(TermPtr, TermPtr),
    /// project a field out of a record
    Proj {
        value: TermPtr,
        field: FieldName,
    },
    /// match variant
    Match {},
    /// cast variant
    Cast {},
    /// lift function into a closure
    FuncAbs {
        target: QualifiedName,
        ty_args: Box<[TermPtr]>,
    },
    /// lift constructor into a closure
    CtorAbs {
        target: QualifiedName,
        ty_args: Box<[TermPtr]>,
    },
    /// Lambda expression
    Lambda {
        binding: UniqueName,
        body: TermPtr,
    },
    /// Let binding
    Let {
        name: UniqueName,
        binding: TermPtr,
        body: TermPtr,
    },
    /// Sequence
    Seq(TermPtr, TermPtr),
    /// Builtin Integer types,
    IntTy(crate::syntax::r#type::Int),
    /// Builtin Float types
    FloatTy(crate::syntax::r#type::Float),
    /// Pi type
    Pi {
        name: UniqueName,
        arg: TermPtr,
        body: TermPtr,
    },
    Var(UniqueName),
    StrTy,
    BooleanTy,
    Universe,
    MetaVar(UniqueName),
    CheckVar,
    Invalid,
}

type Map<K, V> = HashTrieMap<K, V, RcK, FxRandomState>;

impl std::fmt::Display for Term {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Term::Integer(x) => write!(f, "{}", x),
            Term::Float(x) => write!(f, "{}", x),
            Term::Str(x) => write!(f, "{x:?}"),
            Term::Boolean(x) => write!(f, "{x}"),
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
            Term::App(gc, gc1) => {
                write!(f, "({} {})", ***gc, ***gc1)
            }
            Term::Proj { value, field } => todo!(),
            Term::Match {} => todo!(),
            Term::Cast {} => todo!(),
            Term::FuncAbs { target, ty_args } => todo!(),
            Term::CtorAbs { target, ty_args } => todo!(),
            Term::Lambda { binding, body } => {
                write!(f, "λ{}.{}", **binding.0, body.0)
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
                write!(f, "{}", unique_name.0.0)
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
        |value| stacker::maybe_grow(32 * 1024, 1024 * 1024, || crate::sema::term::quote(value));
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
            let var = Rc::new(WithSpan(Value::Var(name.clone()), arg.1));
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
            let var = Rc::new(WithSpan(Value::Var(name.clone()), value.1));
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

#[cfg(test)]
mod test {
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
}
