use chumsky::error::Rich;
use chumsky::extra::Err;
use chumsky::prelude::{choice, just, none_of, one_of};
use chumsky::recursive::recursive;
use chumsky::text::{digits, ident, inline_whitespace, int, newline, whitespace};
use chumsky::{IterParser, Parser};

use crate::print_parse_errors;
use crate::syntax::concrete::{Decl, Def, Expr, File};
use crate::syntax::{Ctor, CtorParams, DataDef, FnDef, FnSig, Ident, Param, fresh};

pub type Msg<'src> = Rich<'src, char>;

macro_rules! out {
    ($o:ty) => { impl Parser<'src, &'src str, $o, Err<Msg<'src>>> };
}

macro_rules! primitive {
    ($kw:literal, $e:ident) => {
        just($kw).map(|_| Expr::$e)
    };
}

#[allow(dead_code)]
pub fn parse(src: &str) -> File {
    file()
        .parse(src)
        .into_result()
        .inspect_err(|es| print_parse_errors(es, src))
        .unwrap()
}

fn file<'src>() -> out!(File<'src>) {
    decl()
        .padded()
        .repeated()
        .collect::<Vec<_>>()
        .map(|decls| File {
            decls: decls.into_boxed_slice(),
        })
}

fn decl<'src>() -> out!(Decl<'src>) {
    fn_decl().or(data_decl())
}

fn fn_decl<'src>() -> out!(Decl<'src>) {
    just("def")
        .padded()
        .ignore_then(identifier())
        .padded()
        .then(typ_params().or_not())
        .padded()
        .then(val_params())
        .padded()
        .then(
            just("->")
                .padded()
                .ignore_then(type_expr())
                .padded()
                .or_not(),
        )
        .then_ignore(just(':'))
        .padded()
        .then(expr())
        .padded_by(inline_whitespace())
        .then_ignore(newline())
        .map(|((((name, typ_params), val_params), ret), body)| Decl {
            name,
            def: Def::Fn(FnDef {
                sig: FnSig {
                    typ_params: typ_params.unwrap_or_default(),
                    val_params,
                    eff: Box::new(Expr::Pure), // TODO: parse effects
                    ret: ret.unwrap_or_else(|| Box::new(Expr::NoneType)),
                },
                body,
            }),
        })
}

fn data_decl<'src>() -> out!(Decl<'src>) {
    just("data")
        .padded()
        .ignore_then(identifier())
        .padded()
        .then(typ_params().or_not())
        .padded()
        .then_ignore(just(':'))
        .then_ignore(inline_whitespace())
        .then_ignore(newline())
        .then(ctors())
        .map(|((name, typ_params), ctors)| Decl {
            name,
            def: Def::Data(DataDef {
                typ_params: typ_params.unwrap_or_default(),
                ctors,
            }),
        })
}

fn ctors<'src>() -> out!(Box<[Ctor<'src, Expr<'src>>]>) {
    ctor()
        .padded_by(inline_whitespace())
        .repeated()
        .at_least(1)
        .collect::<Vec<_>>()
        .map(Vec::into_boxed_slice)
}

fn ctor<'src>() -> out!(Ctor<'src, Expr<'src>>) {
    identifier()
        .then_ignore(inline_whitespace())
        .then(
            ctor_unnamed_params()
                .or(ctor_named_params())
                .or_not()
                .map(Option::unwrap_or_default),
        )
        .then_ignore(inline_whitespace())
        .then_ignore(newline())
        .map(|(name, params)| Ctor { name, params })
}

fn ctor_unnamed_params<'src>() -> out!(CtorParams<'src, Expr<'src>>) {
    just('(')
        .ignore_then(whitespace())
        .ignore_then(
            type_expr()
                .padded()
                .separated_by(just(','))
                .allow_trailing()
                .at_least(1)
                .collect::<Vec<_>>()
                .map(Vec::into_boxed_slice)
                .map(CtorParams::Unnamed),
        )
        .then_ignore(whitespace())
        .then_ignore(just(')'))
}

fn ctor_named_params<'src>() -> out!(CtorParams<'src, Expr<'src>>) {
    just('(')
        .ignore_then(whitespace())
        .ignore_then(
            param()
                .padded()
                .separated_by(just(','))
                .allow_trailing()
                .at_least(1)
                .collect::<Vec<_>>()
                .map(Vec::into_boxed_slice)
                .map(CtorParams::Named),
        )
        .then_ignore(whitespace())
        .then_ignore(just(')'))
}

fn expr<'src>() -> out!(Box<Expr<'src>>) {
    recursive(|value| {
        let lambda = just("lambda")
            .ignore_then(whitespace())
            .ignore_then(
                identifier()
                    .padded()
                    .separated_by(just(','))
                    .collect::<Vec<_>>()
                    .map(Vec::into_boxed_slice),
            )
            .then_ignore(whitespace())
            .then_ignore(just(':'))
            .then_ignore(whitespace())
            .then(value.clone())
            .map(|(params, body)| Expr::Fn { params, body });

        lambda
            .or(primitive!("None", None))
            .or(primitive!("False", True))
            .or(primitive!("True", True))
            .or(str().map(Expr::Str))
            .or(float().map(Expr::Float))
            .or(identifier()
                .then(
                    just('(')
                        .ignore_then(whitespace())
                        .ignore_then(
                            value
                                .padded()
                                .separated_by(just(','))
                                .collect::<Vec<_>>()
                                .map(Vec::into_boxed_slice),
                        )
                        .then_ignore(whitespace())
                        .then_ignore(just(')'))
                        .or_not(),
                )
                .map(|(i, a)| {
                    a.map_or(Expr::Ident(i), |args| Expr::Call {
                        f: Box::new(Expr::Ident(i)),
                        args,
                    })
                }))
            .map(Box::new)
            .boxed()
    })
}

fn str<'src>() -> out!(&'src str) {
    none_of("\\\"")
        .ignored()
        .or(escaped())
        .repeated()
        .to_slice()
        .delimited_by(just('"'), just('"'))
}

fn escaped<'src>() -> out!(()) {
    const INVALID_CHAR: char = '\u{FFFD}';
    just('\\')
        .then(choice((
            just('\\'),
            just('/'),
            just('"'),
            just('b').to('\x08'),
            just('f').to('\x0C'),
            just('n').to('\n'),
            just('r').to('\r'),
            just('t').to('\t'),
            just('u').ignore_then(digits(16).exactly(4).to_slice().validate(
                |digits, e, emitter| {
                    char::from_u32(u32::from_str_radix(digits, 16).unwrap()).unwrap_or_else(|| {
                        emitter.emit(Rich::custom(e.span(), "invalid unicode character"));
                        INVALID_CHAR
                    })
                },
            )),
        )))
        .ignored()
}

fn float<'src>() -> out!(f64) {
    just('-')
        .or_not()
        .then(int(10))
        .then(just('.').then(decimal()).or_not())
        .then(exponential().or_not())
        .to_slice()
        .map(|s: &str| s.parse().unwrap())
}

fn decimal<'src>() -> out!(&'src str) {
    digits(10).to_slice()
}

fn exponential<'src>() -> out!(&'src str) {
    just('e')
        .or(just('E'))
        .then(one_of("+-").or_not())
        .then(decimal())
        .to_slice()
}

fn type_expr<'src>() -> out!(Box<Expr<'src>>) {
    recursive(|value| {
        let fn_type = just('[')
            .ignore_then(whitespace())
            .ignore_then(
                value
                    .clone()
                    .padded()
                    .separated_by(just(','))
                    .collect::<Vec<_>>()
                    .map(Vec::into_boxed_slice),
            )
            .then_ignore(whitespace())
            .then_ignore(just(']'))
            .then_ignore(whitespace())
            .then_ignore(just("->"))
            .then_ignore(whitespace())
            .then(value)
            .map(|(param_types, ret)| {
                Expr::FnType {
                    param_types,
                    eff: Box::new(Expr::Pure), // TODO: parse effects
                    ret,
                }
            });

        fn_type
            .or(primitive!("bool", Boolean))
            .or(primitive!("str", String))
            .or(primitive!("None", NoneType))
            .or(primitive!("f32", F32))
            .or(primitive!("f64", F64))
            .map(Box::new)
            .boxed()
    })
}

fn param<'src>() -> out!(Param<'src, Expr<'src>>) {
    identifier()
        .then_ignore(whitespace())
        .then_ignore(just(':'))
        .then_ignore(whitespace())
        .then(type_expr())
        .map(|(name, typ)| Param { name, typ })
}

fn val_params<'src>() -> out!(Box<[Param<'src, Expr<'src>>]>) {
    just('(')
        .ignore_then(whitespace())
        .ignore_then(
            param()
                .padded()
                .separated_by(just(','))
                .allow_trailing()
                .collect::<Vec<_>>()
                .map(Vec::into_boxed_slice)
                .or_not()
                .map(Option::unwrap_or_default),
        )
        .then_ignore(whitespace())
        .then_ignore(just(')'))
}

fn typ_params<'src>() -> out!(Box<[Param<'src, Expr<'src>>]>) {
    just('[')
        .ignore_then(whitespace())
        .ignore_then(
            identifier()
                .padded()
                .map(|name| Param {
                    name,
                    typ: Box::new(Expr::Type),
                })
                .separated_by(just(','))
                .allow_trailing()
                .at_least(1)
                .collect::<Vec<_>>()
                .map(Vec::into_boxed_slice),
        )
        .then_ignore(whitespace())
        .then_ignore(just(']'))
}

fn identifier<'src>() -> out!(Ident<'src>) {
    ident().map(move |raw| Ident { raw, id: fresh() })
}

#[cfg(test)]
mod tests {
    use crate::syntax::surface::parse;

    #[test]
    fn it_parses_file() {
        parse(
            "

def f0 (  ) -> None : None
def f1 (
) -> None : None

def f2 [
    T,
    U,
] (s: str) -> str :
\"hello, world\"

def f3() -> f64: -114.514

def f4() -> [] -> bool:
    lambda
        :
            True
def f5() -> [bool, str] -> bool: lambda a, b: False

data
  Foo [T]:
    A
B(str)
    C(
        s  : str  ,
        b: bool  ,
    )

",
        );
    }
}
