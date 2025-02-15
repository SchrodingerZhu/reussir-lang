use super::lexer::Token;
use super::r#type::{FieldName, TypePtr, r#type};
use super::{Context, ParserExtra, Ptr, QualifiedName, SmallCollector, WithSpan, map_alloc};
use chumsky::combinator::DelimitedBy;
use chumsky::extra::SimpleState;
use chumsky::input::{MapExtra, ValueInput};
use chumsky::prelude::*;

type ExprPtr<'ctx> = Ptr<'ctx, Expr<'ctx>>;

#[derive(Debug)]
pub enum Expr<'ctx> {
    /// Integer literal
    Int(rug::Integer),
    /// Float literal
    Float(rug::Float),
    /// Boolean literal
    Boolean(bool),
    /// Character literal
    Character(char),
    /// Unit value
    Unit,
    /// String literal
    String(&'ctx str),
    /// Call expression (can be either ctor call, function call or variable call, we do not support direct expr call for now (TODO))
    Call(&'ctx WithSpan<QualifiedName<'ctx>>, &'ctx [ExprPtr<'ctx>]),
    /// If .. {} else {}
    IfThenElse(ExprPtr<'ctx>, ExprPtr<'ctx>, ExprPtr<'ctx>),
    /// Let binding
    Let(WithSpan<&'ctx str>, Option<TypePtr<'ctx>>, ExprPtr<'ctx>),
    /// A sequence of expression
    Sequence(&'ctx [ExprPtr<'ctx>]),
    /// Match expression
    Match(MatchExpr<'ctx>),
    /// Binary expression
    Binary(ExprPtr<'ctx>, WithSpan<BinaryOp>, ExprPtr<'ctx>),
    /// Unary expression
    Unary(WithSpan<UnaryOp>, ExprPtr<'ctx>),
    /// Variable
    Variable(&'ctx str),
}

#[derive(Debug, Clone, Copy, Eq, PartialEq)]
pub enum BinaryOp {
    Add,
    Sub,
    Mod,
    Mul,
    Div,
    LAnd,
    BAnd,
    LOr,
    BOr,
    Xor,
    Shr,
    Shl,
    Eq,
    Ne,
    Le,
    Lt,
    Ge,
    Gt,
}

#[derive(Debug, Clone, Copy, Eq, PartialEq)]
pub enum UnaryOp {
    LNot,
    BNot,
    Neg,
}

/// We don't support nested pattern, literal or range for now.
/// match itself is just a decompose of the record.
#[derive(Debug)]
pub enum Pattern<'ctx> {
    Wildcard,
    Ctor(WithSpan<&'ctx str>, &'ctx [Ptr<'ctx, FieldBinding<'ctx>>]),
}

#[derive(Debug)]
pub struct FieldBinding<'ctx> {
    target: WithSpan<FieldName<'ctx>>,
    rename: Option<WithSpan<&'ctx str>>,
}

#[derive(Debug)]
pub struct MatchExpr<'ctx> {
    value: ExprPtr<'ctx>,
    cases: &'ctx [(Ptr<'ctx, Pattern<'ctx>>, ExprPtr<'ctx>)],
}

macro_rules! expr_parser {

    (standalone $vis:vis $name:ident $body:block) => {
        fn $name<'a, I>() -> impl Parser<'a, I, ExprPtr<'a>, ParserExtra<'a>> + Clone
        where
            I: ValueInput<'a, Token = Token<'a>, Span = SimpleSpan>,
        {
            $body
        }
    };

    (recursive $vis:vis $name:ident $body:expr) => {
        fn $name<'a, I, P>(toplevel: P) -> impl Parser<'a, I, ExprPtr<'a>, ParserExtra<'a>> + Clone
        where
            I: ValueInput<'a, Token = Token<'a>, Span = SimpleSpan>,
            P: Parser<'a, I, ExprPtr<'a>, ParserExtra<'a>> + Clone,
        {
            $body(toplevel)
        }
    };

    () => {};

    ( $vis:vis $name:ident => $body:expr ; $($trailing:tt)* ) => {
        expr_parser!{recursive $vis $name $body}
        expr_parser!{$($trailing)*}
    };

    ( $vis:vis $name:ident -> $body:block ; $($trailing:tt)* ) => {
        expr_parser!{standalone $vis $name $body}
        expr_parser!{$($trailing)*}
    };
}

expr_parser! {
    primitive -> {
        select! {
            Token::Int(x) => Expr::Int(x),
            Token::Float(x) => Expr::Float(x),
            Token::Boolean(x) => Expr::Boolean(x),
            Token::CharLit(x) => Expr::Character(x),
            Token::Unit => Expr::Unit,
            Token::String(x) = m => {
                let state : &&Context<'_> = m.state();
                Expr::String(state.alloc_str(x))
            },
            Token::Ident(x) => Expr::Variable(x),
        }
        .map_with(map_alloc)
    };

    let_expr => | expr | {
        let type_annotation = just(Token::Colon)
            .ignore_then(r#type())
            .repeated()
            .at_most(1)
            .collect::<SmallCollector<_, 1>>()
            .map(|x| x.0.into_iter().next());
        just(Token::Let)
            .ignore_then(select! { Token::Ident(x) => x } . map_with(|x, m| WithSpan(x, m.span())))
            .then(type_annotation)
            .then_ignore(just(Token::Eq))
            .then(expr)
            .map(|((i, a), e)| Expr::Let(i, a, e) )
            .map_with(map_alloc)
    };

    if_then_else_expr => |expr : P| {
        let block = braced_expr_sequence(expr.clone());
        just(Token::If)
            .ignore_then(expr)
            .then(block.clone())
            .then_ignore(just(Token::Else))
            .then(block)
            .map_with(|((a, b), c), m| map_alloc(Expr::IfThenElse(a, b, c), m))
    };

    call_expr => |expr : P| {
        let args = expr
            .separated_by(just(Token::Comma))
            .collect::<SmallCollector<_, 8>>()
            .delimited_by(just(Token::LParen), just(Token::RParen))
            .map_with(|ex, m| {
                let state : &&Context<'_> = m.state();
                state.alloc_slice(ex.0)
            })
            .or(just(Token::Unit).map(|_|{let x : &[ExprPtr] = &[]; x}));
        let name = super::qualified_name();
        name.then(args)
            .map_with(|(name, args), m| {
                map_alloc(Expr::Call(name, args), m)
            })
    };

    pratt_expr => |atom : P| {
        use chumsky::pratt::*;
        let uop = | a, b | just(a).to_span().map(move |s| WithSpan(b, s));
        let bop = | a, b | just(a).to_span().map(move |s| WithSpan(b, s));
        let unary = | a, b, p | prefix(p, uop(a, b), move |op, rhs, m| { map_alloc(Expr::Unary(op, rhs), m) });
        let binary = | a, b, p | infix(right(p), bop(a, b), move |lhs, op, rhs, m| { map_alloc(Expr::Binary(lhs, op, rhs), m) });
        atom.pratt((
            unary(Token::Not, UnaryOp::LNot, 50),
            unary(Token::Tilde, UnaryOp::BNot, 50),
            unary(Token::Minus, UnaryOp::Neg, 50),
            binary(Token::Star, BinaryOp::Mul, 40),
            binary(Token::Slash, BinaryOp::Div, 40),
            binary(Token::Percent, BinaryOp::Mod, 30),
            binary(Token::Plus, BinaryOp::Add, 30),
            binary(Token::Minus, BinaryOp::Sub, 30),
            binary(Token::Shr, BinaryOp::Shr, 25),
            binary(Token::Shl, BinaryOp::Shl, 25),
            binary(Token::And, BinaryOp::BAnd, 24),
            binary(Token::Caret, BinaryOp::Xor, 23),
            binary(Token::Or, BinaryOp::BOr, 22),
            binary(Token::EqEq, BinaryOp::Eq, 20),
            binary(Token::Ne, BinaryOp::Ne, 20),
            binary(Token::Lt, BinaryOp::Lt, 20),
            binary(Token::Le, BinaryOp::Le, 20),
            binary(Token::Gt, BinaryOp::Gt, 20),
            binary(Token::Ge, BinaryOp::Ge, 20),
            binary(Token::AndAnd, BinaryOp::LAnd, 10),
            binary(Token::OrOr, BinaryOp::LOr, 5),
        ))
    };

    parenthesised_expr => |expr : P| {
        expr.delimited_by(just(Token::LParen), just(Token::RParen))
    };

    pub braced_expr_sequence => | expr : P | {
        expr
            .separated_by(just(Token::Semi))
            .collect::<SmallCollector<_, 8>>()
            .delimited_by(just(Token::LBrace), just(Token::RBrace))
            .map_with(|ex, m| {
                let state : &&Context<'_> = m.state();
                if ex.0.len() == 1 {
                    return ex.0.into_iter().next().unwrap();
                }
                let sub_exprs = state.alloc_slice(ex.0);
                map_alloc(Expr::Sequence(sub_exprs), m)
            })
    };

    pub expr -> {
        recursive(|expr| {
            let atom = choice((
                call_expr(expr.clone()),
                primitive(),
                if_then_else_expr(expr.clone()),
                let_expr(expr.clone()),
                braced_expr_sequence(expr.clone()),
                parenthesised_expr(expr),
            ));
            pratt_expr(atom)
        })
    };
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn it_parses_primitive_exprs() {
        let ctx = Context::from_src("123");
        let stream = ctx.token_stream();
        let res = primitive().parse_with_state(stream, &mut &ctx).unwrap();
        println!("{:#?}", res);
        assert!(matches!(res.0, Expr::Int(..)));
    }

    #[test]
    fn it_parses_if_then_else_expr() {
        let ctx = Context::from_src("if true { 123 } else { 456 }");
        let stream = ctx.token_stream();
        let res = expr().parse_with_state(stream, &mut &ctx).unwrap();
        println!("{:#?}", res);
        assert!(matches!(res.0, Expr::IfThenElse(..)));
    }

    #[test]
    fn it_parses_annotated_let_expr() {
        let ctx = Context::from_src("let x : f64 = 1e10");
        let stream = ctx.token_stream();
        let res = expr().parse_with_state(stream, &mut &ctx).unwrap();
        println!("{:#?}", res);
        assert!(matches!(res.0, Expr::Let(..)));
    }

    #[test]
    fn it_parses_simple_let_expr() {
        let ctx = Context::from_src("let x : f64 = 1e10");
        let stream = ctx.token_stream();
        let res = expr().parse_with_state(stream, &mut &ctx).unwrap();
        println!("{:#?}", res);
        assert!(matches!(res.0, Expr::Let(..)));
    }

    #[test]
    fn it_parses_sequence_expr() {
        let ctx = Context::from_src("{let x : f64 = 1e10; 123}");
        let stream = ctx.token_stream();
        let res = expr().parse_with_state(stream, &mut &ctx).unwrap();
        println!("{:#?}", res);
        assert!(matches!(res.0, Expr::Sequence(..)));
    }

    #[test]
    fn it_parses_call_expr() {
        let ctx = Context::from_src("x(1, 2, y, z, std::get(1, 4, \"str\"))");
        let stream = ctx.token_stream();
        let res = expr().parse_with_state(stream, &mut &ctx).unwrap();
        println!("{:#?}", res);
        assert!(matches!(res.0, Expr::Call(..)));
    }

    #[test]
    fn it_parses_unary_expr() {
        let ctx = Context::from_src("~(123)");
        let stream = ctx.token_stream();
        let res = expr().parse_with_state(stream, &mut &ctx).unwrap();
        println!("{:#?}", res);
        assert!(matches!(res.0, Expr::Unary(..)));
    }

    #[test]
    fn it_parses_binary_expr() {
        let ctx = Context::from_src("(1 + ~(123)) / 123 + 123 - 2 * 2 >= -10 - f(-x) && flag");
        let stream = ctx.token_stream();
        let res = expr().parse_with_state(stream, &mut &ctx).unwrap();
        println!("{:#?}", res);
        assert!(matches!(res.0, Expr::Binary(..)));
    }
}
