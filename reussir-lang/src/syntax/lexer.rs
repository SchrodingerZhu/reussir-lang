use chumsky::{
    Parser,
    error::Rich,
    extra::{Full, SimpleState},
    input::{Input, StrInput},
    span::SimpleSpan,
    text::Char,
};

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

pub fn lexer<'ctx, I: StrInput<'ctx, Span = SimpleSpan, Token = char, Slice = &'ctx str>>()
-> impl Parser<'ctx, I, Token<'ctx>, Full<Rich<'ctx, char>, SimpleState<&'ctx Context<'ctx>>, ()>> {
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
        ident => Token::Ident(ident),
    });
    idents_like
}

#[cfg(test)]
mod test {
    use super::*;
    macro_rules! test_single_static {
        ($src:literal, $target:ident) => {
            let context = Context::from_src($src);
            let res = lexer().parse_with_state($src, &mut SimpleState(&context));
            assert_eq!(res.output().unwrap(), &Token::$target);
        };
    }
    #[test]
    fn lexer_recognizes_int_types() {
        test_single_static!("i8", I8);
        test_single_static!("u8", U8);
        test_single_static!("i16", I16);
    }
}
