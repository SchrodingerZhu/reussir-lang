use chumsky::{
    IterParser, Parser,
    container::Container,
    error::Rich,
    extra::{Full, SimpleState},
    input::{Input, StrInput},
    prelude::choice,
    span::SimpleSpan,
    text::{Char, digits},
};
use rug::{Complete, float::Round, ops::CompleteRound};
use smallvec::SmallVec;

use crate::syntax::SmallCollector;

use super::{Context, WithSpan};

#[derive(Debug, Clone, PartialEq)]
pub enum Token<'ctx> {
    // Builtin Types
    I8,
    I16,
    I32,
    I64,
    I128,
    U8,
    U16,
    U32,
    U64,
    U128,
    F16,
    BF16,
    F32,
    F64,
    F128,
    Str,
    Bool,
    Char,
    // Keywords
    If,
    Then,
    Else,
    Fn,
    Return,
    Yield,
    Region,
    Unboxed,
    Data,
    Match,
    True,
    False,
    Let,
    // Delimiters
    LParen,
    RParen,
    LSquare,
    RSquare,
    LBrace,
    RBrace,
    // Operators,
    Plus,
    Minus,
    Star,
    Flash,
    Percent,
    Caret,
    Not,
    And,
    Or,
    AndAnd,
    OrOr,
    Shl,
    Shr,
    Eq,
    EqEq,
    Ne,
    Gt,
    Lt,
    Ge,
    Le,
    At,
    Dot,
    Comma,
    Semi,
    Colon,
    PathSep,
    RArrow,
    FatArrow,
    // Literals and Identifiers
    Int(rug::Integer),
    Float(rug::Float),
    String(String),
    Ident(&'ctx str),
    CharLit(char),
}

type LexerExtra<'ctx> = Full<Rich<'ctx, char>, SimpleState<&'ctx Context<'ctx>>, ()>;

pub fn integer_literal_lexer<'ctx>() -> impl Parser<'ctx, &'ctx str, rug::Integer, LexerExtra<'ctx>>
{
    use chumsky::primitive::*;
    fn integer_literal_with_radix<'ctx, const PREFIX: char, const RADIX: u32>()
    -> impl Parser<'ctx, &'ctx str, rug::Integer, LexerExtra<'ctx>> {
        use chumsky::prelude::*;
        let body = chumsky::text::digits(RADIX)
            .separated_by(just('_'))
            .to_slice()
            .try_map(|s: &str, span: SimpleSpan| {
                rug::Integer::parse_radix(s, RADIX as i32)
                    .map(|x| x.complete())
                    .map_err(|e| Rich::custom(span, format!("failed to parse integer body {e}")))
            });

        let prefix = just('0').ignore_then(just(PREFIX));
        prefix.ignore_then(body)
    }
    fn decimal_integer_literal<'ctx>()
    -> impl Parser<'ctx, &'ctx str, rug::Integer, LexerExtra<'ctx>> {
        use chumsky::prelude::*;
        chumsky::text::digits(10)
            .separated_by(just('_'))
            .to_slice()
            .try_map(|s: &str, span: SimpleSpan| {
                rug::Integer::parse(s)
                    .map(|x| x.complete())
                    .map_err(|e| Rich::custom(span, format!("failed to parse integer body {e}")))
            })
    }
    let body = choice((
        integer_literal_with_radix::<'b', 2>(),
        integer_literal_with_radix::<'o', 8>(),
        integer_literal_with_radix::<'x', 16>(),
        decimal_integer_literal(),
    ));
    one_of(['+', '-'])
        .repeated()
        .at_most(1)
        .collect::<SmallCollector<char, 1>>()
        .then(body)
        .map(|(op, body)| {
            if op.0.is_empty() || op.0[0] == '+' {
                body
            } else {
                -body
            }
        })
}

pub fn float_literal_lexer<'ctx>() -> impl Parser<'ctx, &'ctx str, rug::Float, LexerExtra<'ctx>> {
    use chumsky::prelude::*;
    let optional_sign = one_of(['+', '-']).repeated().at_most(1);
    let special = choice((just("nan"), just("inf"))).ignored();
    let exponent = one_of(['e', 'E'])
        .then(optional_sign)
        .then(digits(10))
        .ignored();
    let a_exp = digits(10).then(exponent).ignored();
    let a_dot_b_exp = digits(10)
        .then(just('.'))
        .then(digits(10))
        .then(exponent.repeated().at_most(1))
        .ignored();
    let body = choice((special, a_exp, a_dot_b_exp));
    optional_sign
        .ignore_then(body)
        .to_slice()
        .try_map(|src, span| {
            rug::Float::parse(src)
                .map_err(|e| Rich::custom(span, "cannot parse floating point literal {e}"))
                .map(|x| x.complete(128))
        })
}

pub fn unqouted_char_literal_lexer<'ctx, const QUOTE: char>()
-> impl Parser<'ctx, &'ctx str, char, LexerExtra<'ctx>> {
    use chumsky::prelude::*;
    let unicode_digits = digits(16)
        .at_most(6)
        .to_slice()
        .delimited_by(just('{'), just('}'))
        .try_map(|x, span| {
            u32::from_str_radix(x, 16)
                .map_err(|e| {
                    Rich::custom(
                        span,
                        format!("failed to convert unicode sequence as U32: {e}"),
                    )
                })
                .and_then(|x| {
                    char::from_u32(x)
                        .ok_or_else(|| Rich::custom(span, "unicode value out of scope".to_string()))
                })
        });
    let unicode_seq = just('u').ignore_then(unicode_digits);
    let ch = |x: char, y: char| just(x).to(y);
    let escaped_char = just('\\').ignore_then(choice((
        unicode_seq,
        ch('n', '\n'),
        ch('r', '\r'),
        ch('t', '\t'),
        ch('b', '\u{08}'),
        ch('f', '\u{0C}'),
        ch('\\', '\\'),
        ch('\'', '\''),
        ch('"', '"'),
    )));
    escaped_char.or(any().and_is(just(QUOTE).or(just('\\')).not()))
}

pub fn lexer<'ctx>() -> impl Parser<'ctx, &'ctx str, Token<'ctx>, LexerExtra<'ctx>> {
    use chumsky::prelude::*;
    let idents_like = chumsky::text::ident().map(|ident: &str| match ident {
        "i8" => Token::I8,
        "i16" => Token::I16,
        "i32" => Token::I32,
        "i64" => Token::I64,
        "i128" => Token::I128,
        "u8" => Token::U8,
        "u16" => Token::U16,
        "u32" => Token::U32,
        "u64" => Token::U64,
        "u128" => Token::U128,
        "f16" => Token::F16,
        "bf16" => Token::BF16,
        "f32" => Token::F32,
        "f64" => Token::F64,
        "f128" => Token::F128,
        "str" => Token::Str,
        "bool" => Token::Bool,
        "char" => Token::Char,
        ident => Token::Ident(ident),
    });
    let integer = integer_literal_lexer().map(Token::Int);
    let float = float_literal_lexer().map(Token::Float);
    let char_lit = unqouted_char_literal_lexer::<'\''>()
        .delimited_by(just('\''), just('\''))
        .map(Token::CharLit);
    let string_lit = unqouted_char_literal_lexer::<'\"'>()
        .repeated()
        .collect::<String>()
        .delimited_by(just('\"'), just('\"'))
        .map(Token::String);
    choice((float, integer, idents_like, char_lit, string_lit))
}

#[cfg(test)]
mod test {
    use super::*;
    macro_rules! test_single_static {
        ($src:literal, $target:ident) => {
            let context = Context::from_src($src);
            let res = lexer().parse_with_state($src, &mut SimpleState(&context));
            println!("{res:#?}");
            assert_eq!(res.output().unwrap(), &Token::$target);
        };
        ($src:literal, $target:ident, $value:expr) => {
            let context = Context::from_src($src);
            let res = lexer().parse_with_state($src, &mut SimpleState(&context));
            println!("{res:#?}");
            assert_eq!(res.output().unwrap(), &Token::$target($value));
        };
        ($src:literal, $target:ident => $lambda:expr) => {
            let context = Context::from_src($src);
            let res = lexer().parse_with_state($src, &mut SimpleState(&context));
            println!("{res:#?}");
            match res.output().unwrap() {
                Token::$target(x) => assert!($lambda(x)),
                _ => panic!("unexpected token kind"),
            }
        };
    }

    #[test]
    fn lexer_recognizes_int_types() {
        test_single_static!("i8", I8);
        test_single_static!("u8", U8);
        test_single_static!("i16", I16);
    }

    #[test]
    fn lexer_recognizes_char_literals() {
        test_single_static!(r"'a'", CharLit, 'a');
        test_single_static!(r"'\n'", CharLit, '\n');
        test_single_static!(r"'\''", CharLit, '\'');
        test_single_static!(r"'\\'", CharLit, '\\');
        test_single_static!(r"'😊'", CharLit, '😊');
        test_single_static!(r"'\u{263A}'", CharLit, '☺');
    }
    #[test]
    fn lexer_recognizes_string_literals() {
        test_single_static!(r#""abc""#, String, "abc".to_string());
        test_single_static!(r#""abc\"""#, String, "abc\"".to_string());
        test_single_static!(
            r#""😊😊😊123$$\n\t""#,
            String,
            "😊😊😊123$$\n\t".to_string()
        );
    }

    #[test]
    fn lexer_recognizes_int_literals() {
        test_single_static!("123", Int, "123".parse().unwrap());
        test_single_static!("-123", Int, "-123".parse().unwrap());
        test_single_static!("123_456_789", Int, "123_456_789".parse().unwrap());
        test_single_static!(
            "0x123_456_789_ABC",
            Int,
            rug::Integer::from(0x123456789ABCi64)
        );
        test_single_static!(
            "-0x123_456_789_ABC",
            Int,
            rug::Integer::from(-0x123456789ABCi64)
        );
        test_single_static!(
            "0x123456789ABCABCABCABCDEFDEFDEFDEFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF",
            Int,
            rug::Integer::from_str_radix(
                "123456789ABCABCABCABCDEFDEFDEFDEFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF",
                16
            )
            .unwrap()
        );
    }

    #[test]
    fn lexer_recognizes_special_floats() {
        test_single_static!("nan", Float => |x : &rug::Float| x.is_nan());
        test_single_static!("inf", Float => |x : &rug::Float| x.is_infinite());
    }

    #[test]
    fn lexer_recognizes_float_literals() {
        test_single_static!(
            "12.0",
            Float,
            rug::Float::parse("12.0").unwrap().complete(128)
        );
        test_single_static!(
            "-12.12345",
            Float,
            rug::Float::parse("-12.12345").unwrap().complete(128)
        );
        test_single_static!(
            "-12E12345",
            Float,
            rug::Float::parse("-12E12345").unwrap().complete(128)
        );
        test_single_static!(
            "+12.123E12345",
            Float,
            rug::Float::parse("+12.123E12345").unwrap().complete(128)
        );
        test_single_static!(
            "+12.123E-12345",
            Float,
            rug::Float::parse("+12.123E-12345").unwrap().complete(128)
        );
    }
}
