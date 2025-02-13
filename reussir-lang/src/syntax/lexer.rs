use std::convert::identity;

use super::{Context, WithSpan};
use chumsky::{
    error::Rich,
    input::{Input, Stream, ValueInput},
    span::SimpleSpan,
};
use logos::{Lexer, Logos};
use unescaper::unescape;

#[derive(thiserror::Error, Debug, Clone, Default, PartialEq)]
pub enum Error {
    #[default]
    #[error("Unknown token encountered")]
    UnknownToken,
    #[error("failed to parse integer: {0}")]
    InvalidIntLiteral(#[from] rug::integer::ParseIntegerError),
    #[error("failed to parse floating point: {0}")]
    InvalidFloatLiteral(#[from] rug::float::ParseFloatError),
    #[error("invalid escape sequence: {0}")]
    InvalidEscapeSequence(String),
    #[error("non-single character")]
    NonSingleCharater(String),
}

#[derive(Debug, Clone, PartialEq, Logos)]
#[logos(error = Error)]
#[logos(skip r"//[^\n]*")]
#[logos(skip r"[ \t\r\n\f]+")]
#[logos(skip r"/\*(?:[^*]|\*[^/])*\*/")]
pub enum Token<'src> {
    // Builtin Types
    #[token("i8")]
    I8,
    #[token("i16")]
    I16,
    #[token("i32")]
    I32,
    #[token("i64")]
    I64,
    #[token("i128")]
    I128,
    #[token("u8")]
    U8,
    #[token("u16")]
    U16,
    #[token("u32")]
    U32,
    #[token("u64")]
    U64,
    #[token("u128")]
    U128,
    #[token("f16")]
    F16,
    #[token("bf16")]
    BF16,
    #[token("f32")]
    F32,
    #[token("f64")]
    F64,
    #[token("f128")]
    F128,
    #[token("str")]
    Str,
    #[token("bool")]
    Bool,
    #[token("char")]
    Char,
    // Keywords
    #[token("if")]
    If,
    #[token("then")]
    Then,
    #[token("else")]
    Else,
    #[token("fn")]
    Fn,
    #[token("return")]
    Return,
    #[token("yield")]
    Yield,
    #[token("cond")]
    Cond,
    #[token("region")]
    Region,
    #[token("data")]
    Data,
    #[token("match")]
    Match,
    #[token("let")]
    Let,
    // Delimiters
    #[token("(")]
    LParen,
    #[token(")")]
    RParen,
    #[token("[")]
    LSquare,
    #[token("]")]
    RSquare,
    #[token("{")]
    LBrace,
    #[token("}")]
    RBrace,
    // Operators,
    #[token("+")]
    Plus,
    #[token("-")]
    Minus,
    #[token("*")]
    Star,
    #[token("/")]
    Slash,
    #[token("%")]
    Percent,
    #[token("^")]
    Caret,
    #[token("!")]
    Not,
    #[token("~")]
    Tilde,
    #[token("&")]
    And,
    #[token("|")]
    Or,
    #[token("&&")]
    AndAnd,
    #[token("||")]
    OrOr,
    #[token("<<")]
    Shl,
    #[token(">>")]
    Shr,
    #[token("=")]
    Eq,
    #[token("==")]
    EqEq,
    #[token("!=")]
    Ne,
    #[token(">")]
    Gt,
    #[token("<")]
    Lt,
    #[token(">=")]
    Ge,
    #[token("<=")]
    Le,
    #[token("@")]
    At,
    #[token(".")]
    Dot,
    #[token(",")]
    Comma,
    #[token(";")]
    Semi,
    #[token(":")]
    Colon,
    #[token("::")]
    PathSep,
    #[token("->")]
    RArrow,
    #[token("<-")]
    LArrow,
    #[token("=>")]
    FatArrow,
    #[token("#")]
    Sharp,
    #[token("()")]
    Unit,
    // Literals and Identifiers
    #[regex(r"[+\-]?\d[\d_]*", decimal_integer_callback)]
    #[regex(r"[+\-]?0x[\da-fA-F_]+", radix_integer_callback::<16>)]
    #[regex(r"[+\-]?0o[0-8_]+", radix_integer_callback::<8>)]
    #[regex(r"[+\-]?0b[01_]+", radix_integer_callback::<2>)]
    Int(rug::Integer),
    #[regex(
        r"[+\-]?([\d]+(\.\d*)|[\d]+(\.\d*)?([eE][+\-]?\d+))",
        float_literal_callback
    )]
    #[regex(r"[+\-]?inf", float_literal_callback)]
    #[regex(r"[+\-]?nan", float_literal_callback)]
    Float(rug::Float),
    #[regex(r#""(?:[^"]|\\")*""#, string_literal_callback)]
    String(String),
    #[regex(r#"'(?:[^']|\\')*'"#, char_literal_callback)]
    CharLit(char),
    #[token("true", |_| true)]
    #[token("false", |_| false)]
    Boolean(bool),
    #[regex(r"\p{XID_Start}\p{XID_Continue}*")]
    Ident(&'src str),
    Error(Box<Error>),
}

fn decimal_integer_callback<'a>(lex: &mut Lexer<'a, Token<'a>>) -> Result<rug::Integer, Error> {
    lex.slice().parse().map_err(Into::into)
}

fn float_literal_callback<'a>(lex: &mut Lexer<'a, Token<'a>>) -> Result<rug::Float, Error> {
    use rug::ops::CompleteRound;
    rug::Float::parse(lex.slice())
        .map_err(Into::into)
        .map(|x| x.complete(128))
}

fn string_literal_callback<'a>(lex: &mut Lexer<'a, Token<'a>>) -> Result<String, Error> {
    let slice = lex.slice();
    let slice = &slice[1..slice.len() - 1];
    unescape(slice).map_err(|x| Error::InvalidEscapeSequence(format!("{x}")))
}

fn char_literal_callback<'a>(lex: &mut Lexer<'a, Token<'a>>) -> Result<char, Error> {
    let slice = lex.slice();
    let slice = &slice[1..slice.len() - 1];
    unescape(slice)
        .map_err(|x| Error::InvalidEscapeSequence(format!("{x}")))
        .and_then(|x| {
            let mut chars = x.chars();
            match chars.next() {
                Some(x) if chars.next().is_none() => Ok(x),
                _ => Err(Error::NonSingleCharater(x)),
            }
        })
}

fn radix_integer_callback<'a, const RADIX: usize>(
    lex: &mut Lexer<'a, Token<'a>>,
) -> Result<rug::Integer, Error> {
    let slice = lex.slice();
    let skip = if slice.starts_with('+') || slice.starts_with('-') {
        3
    } else {
        2
    };
    rug::Integer::parse_radix(&slice[skip..], RADIX as i32)
        .map_err(Into::into)
        .map(Into::into)
        .map(|x: rug::Integer| if slice.starts_with('-') { -x } else { x })
}

impl Context<'_> {
    pub fn token_stream(&self) -> impl ValueInput<Token = Token, Span = SimpleSpan> {
        let iter = Token::lexer(&self.src)
            .spanned()
            .map(|(res, range)| match res {
                Ok(tk) => (tk, range.into()),
                Err(err) => (Token::Error(Box::new(err)), range.into()),
            });
        Stream::from_iter(iter).map((0..self.src.len()).into(), identity)
    }
}

#[cfg(test)]
mod test {
    use rug::az::UnwrappedAs;

    use super::*;
    macro_rules! test_single_static {
        ($src:literal, $target:ident) => {
            let res = Token::lexer($src).next().unwrap().unwrap();
            println!("{res:#?}");
            assert_eq!(res, Token::$target);
        };
        ($src:literal, $target:ident, $value:expr) => {
            let res = Token::lexer($src).next().unwrap().unwrap();
            println!("{res:#?}");
            assert_eq!(res, Token::$target($value.into()));
        };
        ($src:literal, $target:ident => $lambda:expr) => {
            let res = Token::lexer($src).next().unwrap().unwrap();
            println!("{res:#?}");
            match res {
                Token::$target(ref x) => assert!($lambda(x)),
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
        test_single_static!(r#""abc""#, String, "abc");
        test_single_static!(r#""abc\"""#, String, "abc\"");
        test_single_static!(r#""😊😊😊123$$\n\t""#, String, "😊😊😊123$$\n\t");
    }

    #[test]
    fn lexer_recognizes_int_literals() {
        test_single_static!("123", Int, rug::Integer::parse("123").unwrap());
        test_single_static!("-123", Int, rug::Integer::parse("-123").unwrap());
        test_single_static!(
            "123_456_789",
            Int,
            rug::Integer::parse("123_456_789").unwrap()
        );
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
        use rug::ops::CompleteRound;
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

    #[test]
    fn lexer_tokenizes_simple_input() {
        const SRC: &str = r#"
    data Rbtree = Leaf | Branch(RbTree, i32, RbTree);
    // Get value of a red black tree
    #[inline]
    fn value(t: RbTree) -> i32 {
      match t {
        Leaf => 0, /* Return 0 if not defined */
        Branch(l, x, r) => x,
      }
    }
    #[inline]
    fn foo() -> f64 {
      3.1415926535
    }
"#;
        let stream = Token::lexer(SRC).spanned();
        for (i, s) in stream {
            println!("{:?}", (i.unwrap(), s));
        }
    }
}
