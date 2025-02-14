use super::lexer::Token;
use super::r#type::TypePtr;
use super::{Context, ParserExtra, WithSpan, map_alloc};
use chumsky::combinator::DelimitedBy;
use chumsky::extra::SimpleState;
use chumsky::input::{MapExtra, ValueInput};
use chumsky::prelude::*;
type ExprPtr<'ctx> = &'ctx WithSpan<Expr<'ctx>>;

#[derive(Debug)]
pub enum Expr<'ctx> {
    Int(rug::Integer),
    Float(rug::Float),
    Boolean(bool),
    Character(char),
    /// String literal
    String(&'ctx str),
    IfThenElse(ExprPtr<'ctx>, ExprPtr<'ctx>, ExprPtr<'ctx>),
    Let(WithSpan<&'ctx str>, Option<TypePtr<'ctx>>, ExprPtr<'ctx>),
    /// Semicolon separated exprs
    Sequence(&'ctx [ExprPtr<'ctx>]),
}

macro_rules! expr_parser {
    ($($name:ident => $body:block)+) => {
        $(fn $name<'a, I>() -> impl Parser<'a, I, ExprPtr<'a>, ParserExtra<'a>> + Clone
        where
            I: ValueInput<'a, Token = Token<'a>, Span = SimpleSpan>,
        $body)+
    };
}

expr_parser! {
    primitive => {
        select! {
            Token::Int(x) => Expr::Int(x),
            Token::Float(x) => Expr::Float(x),
            Token::Boolean(x) => Expr::Boolean(x),
            Token::CharLit(x) => Expr::Character(x),
            Token::String(x) = m => {
                let state : &&Context<'_> = m.state();
                Expr::String(state.alloc_str(x))
            }
        }
        .map_with(map_alloc)
    }

    expr => {
        recursive(|expr| {
            let braced_expr = || {
                expr.clone()
                .delimited_by(just(Token::LBrace), just(Token::RBrace))
            };
            let if_then_else = just(Token::If)
                .ignore_then(expr.clone())
                .then(braced_expr())
                .then_ignore(just(Token::Else))
                .then(braced_expr())
                .map_with(|((a, b), c), m| map_alloc(Expr::IfThenElse(a, b, c), m));
            choice(
              (
                  primitive(),
                  if_then_else,
              )
            )
        })
    }
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn it_parses_primitive_exprs() {
        let ctx = Context::from_src("123");
        let stream = ctx.token_stream();
        let res = primitive().parse_with_state(stream, &mut SimpleState(&ctx));
        println!("{:#?}", res.unwrap())
    }

    #[test]
    fn it_parses_if_then_else() {
        let ctx = Context::from_src("if true { 123 } else { 456 }");
        let stream = ctx.token_stream();
        let res = expr().parse_with_state(stream, &mut SimpleState(&ctx));
        println!("{:#?}", res.unwrap())
    }
}
