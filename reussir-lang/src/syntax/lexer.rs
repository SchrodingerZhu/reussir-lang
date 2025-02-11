use chumsky::{
    IterParser, Parser,
    container::Container,
    error::Rich,
    extra::{Full, SimpleState},
    input::{Input, StrInput},
    span::SimpleSpan,
    text::Char,
};
use rug::Complete;
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
    String(&'ctx str),
    Ident(&'ctx str),
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

pub fn lexer<'ctx>() -> impl Parser<'ctx, &'ctx str, Token<'ctx>, LexerExtra<'ctx>> {
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
        ident => Token::Ident(ident),
    });
    let integer = integer_literal_lexer().map(Token::Int);
    idents_like.or(integer)
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
    }
    #[test]
    fn lexer_recognizes_int_types() {
        test_single_static!("i8", I8);
        test_single_static!("u8", U8);
        test_single_static!("i16", I16);
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
}
