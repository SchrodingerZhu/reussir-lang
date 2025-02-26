use super::lexer::Token;
use super::stmt::FuncParam;
use super::r#type::{TypePtr, r#type};
use super::{
    Attribute, Context, FieldName, ParserExtra, Ptr, QualifiedName, SmallCollector, WithSpan,
    attribute, map_alloc, qualified_name, spanned_ident,
};
use chumsky::combinator::DelimitedBy;
use chumsky::extra::SimpleState;
use chumsky::input::{MapExtra, ValueInput};
use chumsky::prelude::*;

pub type ExprPtr<'ctx> = Ptr<'ctx, Expr<'ctx>>;

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
    Call(Ptr<'ctx, QualifiedName<'ctx>>, &'ctx [ExprPtr<'ctx>]),
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
    Variable(Ptr<'ctx, QualifiedName<'ctx>>),
    /// Attributed expression (future-proof)
    Attributed(Ptr<'ctx, Attribute<'ctx>>, ExprPtr<'ctx>),
    /// Freeze region
    Freeze(ExprPtr<'ctx>),
    /// Lambda expression
    Lambda {
        params: &'ctx [Ptr<'ctx, LambdaParam<'ctx>>],
        ret: Option<TypePtr<'ctx>>,
        body: ExprPtr<'ctx>,
    },
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
    Ctor {
        name: Ptr<'ctx, QualifiedName<'ctx>>,
        bindings: &'ctx [Ptr<'ctx, FieldBinding<'ctx>>],
        discard: bool,
    },
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

fn unamed_bindings<'a, I>()
-> impl Parser<'a, I, &'a [Ptr<'a, FieldBinding<'a>>], ParserExtra<'a>> + Clone
where
    I: ValueInput<'a, Token = Token<'a>, Span = SimpleSpan>,
{
    spanned_ident()
        .separated_by(just(Token::Comma))
        .allow_trailing()
        .collect::<SmallCollector<_, 4>>()
        .map_with(|fields, m| -> &[&WithSpan<FieldBinding<'_>>] {
            let state: &&Context = m.state();
            let iter = fields.0.into_iter().enumerate().map(|(idx, rename)| {
                let span = rename.1;
                let binding = FieldBinding {
                    target: WithSpan(FieldName::Idx(idx), span),
                    rename: Some(rename),
                };
                state.alloc(WithSpan(binding, span))
            });
            state.alloc_slice(iter)
        })
}

fn named_bindings<'a, I>()
-> impl Parser<'a, I, &'a [Ptr<'a, FieldBinding<'a>>], ParserExtra<'a>> + Clone
where
    I: ValueInput<'a, Token = Token<'a>, Span = SimpleSpan>,
{
    let ident = spanned_ident();
    let rename = just(Token::Colon)
        .ignore_then(ident)
        .repeated()
        .at_most(1)
        .collect::<SmallCollector<_, 1>>()
        .map(|rename| rename.0.into_iter().next());
    let binding = ident
        .map(|x| WithSpan(FieldName::Name(x.0), x.1))
        .then(rename)
        .map_with(|(target, rename), m| map_alloc(FieldBinding { target, rename }, m));
    binding
        .separated_by(just(Token::Comma))
        .allow_trailing()
        .collect::<SmallCollector<_, 4>>()
        .map_with(|fields, m| {
            let state: &&Context = m.state();
            state.alloc_slice(fields.0)
        })
}

fn ctor_pattern<'a, I>() -> impl Parser<'a, I, Ptr<'a, Pattern<'a>>, ParserExtra<'a>> + Clone
where
    I: ValueInput<'a, Token = Token<'a>, Span = SimpleSpan>,
{
    let discard = just(Token::DotDot)
        .repeated()
        .at_most(1)
        .count()
        .map(|cnt| cnt != 0);
    let named_bindings = named_bindings()
        .then(discard.clone())
        .delimited_by(just(Token::LBrace), just(Token::RBrace));
    let unamed_bindings = unamed_bindings()
        .then(discard)
        .delimited_by(just(Token::LParen), just(Token::RParen));
    let bindings = named_bindings
        .or(unamed_bindings)
        .repeated()
        .at_most(1)
        .collect::<SmallCollector<_, 1>>()
        .map(|c| c.0.into_iter().next().unwrap_or((&[], false)));
    qualified_name()
        .then(bindings)
        .map_with(|(name, (bindings, discard)), m| {
            let state: &&Context = m.state();
            let ctor = Pattern::Ctor {
                name,
                bindings,
                discard,
            };
            map_alloc(ctor, m)
        })
}

fn pattern<'a, I>() -> impl Parser<'a, I, Ptr<'a, Pattern<'a>>, ParserExtra<'a>> + Clone
where
    I: ValueInput<'a, Token = Token<'a>, Span = SimpleSpan>,
{
    just(Token::Underscore)
        .to_span()
        .map_with(|span, m| map_alloc(Pattern::Wildcard, m))
        .or(ctor_pattern())
}

macro_rules! expr_parser {

    (standalone $vis:vis $name:ident $body:block) => {
        $vis fn $name<'a, I>() -> impl Parser<'a, I, ExprPtr<'a>, ParserExtra<'a>> + Clone
        where
            I: ValueInput<'a, Token = Token<'a>, Span = SimpleSpan>,
        {
            $body
        }
    };

    (recursive $vis:vis $name:ident $body:expr) => {
        $vis fn $name<'a, I, P>(toplevel: P) -> impl Parser<'a, I, ExprPtr<'a>, ParserExtra<'a>> + Clone
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

fn attributed_expr<'a, I, A, P>(
    toplevel: P,
    attr: A,
) -> impl Parser<'a, I, ExprPtr<'a>, ParserExtra<'a>> + Clone
where
    I: ValueInput<'a, Token = Token<'a>, Span = SimpleSpan>,
    P: Parser<'a, I, ExprPtr<'a>, ParserExtra<'a>> + Clone,
    A: Parser<'a, I, Ptr<'a, Attribute<'a>>, ParserExtra<'a>> + Clone,
{
    attr.then(toplevel)
        .map(|(a, e)| Expr::Attributed(a, e))
        .map_with(map_alloc)
}

#[derive(Debug)]
pub struct LambdaParam<'ctx> {
    name: WithSpan<&'ctx str>,
    r#type: Option<TypePtr<'ctx>>,
}

fn lambda_params<'a, I, T>(
    opt_type: T,
) -> impl Parser<'a, I, &'a [Ptr<'a, LambdaParam<'a>>], ParserExtra<'a>> + Clone
where
    I: ValueInput<'a, Token = Token<'a>, Span = SimpleSpan>,
    T: Parser<'a, I, Option<TypePtr<'a>>, ParserExtra<'a>> + Clone,
{
    spanned_ident()
        .then(opt_type)
        .map(|(name, r#type)| LambdaParam { name, r#type })
        .map_with(map_alloc)
        .separated_by(just(Token::Comma))
        .collect::<SmallCollector<_, 4>>()
        .map_with(|c, m| {
            let state: &&Context = m.state();
            state.alloc_slice(c.0)
        })
        .delimited_by(just(Token::Or), just(Token::Or))
}

fn lambda_expr<'a, I, P>(toplevel: P) -> impl Parser<'a, I, ExprPtr<'a>, ParserExtra<'a>> + Clone
where
    I: ValueInput<'a, Token = Token<'a>, Span = SimpleSpan>,
    P: Parser<'a, I, ExprPtr<'a>, ParserExtra<'a>> + Clone,
{
    let r#type = super::r#type::r#type();
    let opt_type = just(Token::Colon)
        .ignore_then(r#type)
        .repeated()
        .at_most(1)
        .collect::<SmallCollector<_, 1>>()
        .map(|c| c.0.into_iter().next());
    let params = lambda_params(opt_type.clone());
    params
        .then(opt_type)
        .then(toplevel)
        .map(|((params, ret), body)| Expr::Lambda { params, ret, body })
        .map_with(map_alloc)
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
                let state : &&Context = m.state();
                Expr::String(state.alloc_str(x))
            },
        }
        .map_with(map_alloc)
    };

    variable -> {
        qualified_name().map(Expr::Variable).map_with(map_alloc)
    };

    let_expr => | expr | {
        let type_annotation = just(Token::Colon)
            .ignore_then(r#type())
            .repeated()
            .at_most(1)
            .collect::<SmallCollector<_, 1>>()
            .map(|x| x.0.into_iter().next());
        just(Token::Let)
            .ignore_then(spanned_ident())
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
                let state : &&Context = m.state();
                state.alloc_slice(ex.0)
            })
            .or(just(Token::Unit).map(|_|{let x : &[ExprPtr] = &[]; x}));
        let name = super::qualified_name();
        name.then(args)
            .map_with(|(name, args), m| {
                map_alloc(Expr::Call(name, args), m)
            })
    };

    freeze_expr => |expr : P| {
        just(Token::Freeze)
            .ignore_then(expr)
            .map(Expr::Freeze)
            .map_with(map_alloc)
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

    match_expr => |expr : P| {
        let cases = pattern()
            .then_ignore(just(Token::FatArrow))
            .then(expr.clone())
            .separated_by(just(Token::Comma))
            .collect::<SmallCollector<_, 8>>()
            .delimited_by(just(Token::LBrace), just(Token::RBrace))
            .map_with(|c, m| {
                let state : &&Context = m.state();
                state.alloc_slice(c.0)
            });
        just(Token::Match)
            .ignore_then(expr.clone())
            .then(cases)
            .map_with(|(value, cases), m| map_alloc(Expr::Match(MatchExpr{value, cases}), m))
    };

    pub braced_expr_sequence => | expr : P | {
        expr
            .separated_by(just(Token::Semi))
            .collect::<SmallCollector<_, 8>>()
            .delimited_by(just(Token::LBrace), just(Token::RBrace))
            .map_with(|ex, m| {
                let state : &&Context = m.state();
                if ex.0.len() == 1 {
                    return ex.0.into_iter().next().unwrap();
                }
                let sub_exprs = state.alloc_slice(ex.0);
                map_alloc(Expr::Sequence(sub_exprs), m)
            })
    };

    pub expr -> {
        recursive(|expr| {
            let attr = attribute(expr.clone());
            let atom = choice((
                freeze_expr(expr.clone()),
                call_expr(expr.clone()),
                primitive(),
                variable(),
                if_then_else_expr(expr.clone()),
                let_expr(expr.clone()),
                braced_expr_sequence(expr.clone()),
                match_expr(expr.clone()),
                attributed_expr(expr.clone(), attr.clone()),
                lambda_expr(expr.clone()),
                parenthesised_expr(expr),
            ));
            pratt_expr(atom)
        })
    };
}

#[cfg(test)]
mod test {
    use super::*;

    macro_rules! test_expr_parser {
        ($name:ident, $src:literal, $parser:ident, $pattern:pat) => {
            #[test]
            fn $name() {
                let ctx = Context::from_src($src);
                let stream = ctx.token_stream();
                let res = $parser().parse_with_state(stream, &mut &ctx).unwrap();
                println!("{:#?}", res);
                assert!(matches!(res.0, $pattern));
            }
        };
    }

    test_expr_parser!(it_parses_primitive_exprs, "123", primitive, Expr::Int(..));

    test_expr_parser!(
        it_parses_if_then_else_expr,
        "if true { 123 } else { 456 }",
        expr,
        Expr::IfThenElse(..)
    );

    test_expr_parser!(
        it_parses_annotated_let_expr,
        "let x : f64 = 1e10",
        expr,
        Expr::Let(..)
    );

    test_expr_parser!(
        it_parses_simple_let_expr,
        "let x : f64 = 1e10",
        expr,
        Expr::Let(..)
    );

    test_expr_parser!(
        it_parses_sequence_expr,
        "{let x : f64 = 1e10; 123}",
        expr,
        Expr::Sequence(..)
    );

    test_expr_parser!(
        it_parses_call_expr,
        r#"x(1, 2, y, z, std::get(1, 4, "str"))"#,
        expr,
        Expr::Call(..)
    );

    test_expr_parser!(it_parses_unary_expr, r#"~(123)"#, expr, Expr::Unary(..));

    test_expr_parser!(
        it_parses_binary_expr,
        "(1 + ~(123)) / 123 + 123 - 2 * 2 >= -10 - f(-x) && flag",
        expr,
        Expr::Binary(..)
    );

    test_expr_parser!(it_parses_wildcard_pattern, "_", pattern, Pattern::Wildcard);

    test_expr_parser!(
        it_parses_singleton_pattern,
        "std::ds::A",
        pattern,
        Pattern::Ctor { .. }
    );

    test_expr_parser!(
        it_parses_braced_ctor_pattern,
        "Cons { head, tail : tl }",
        pattern,
        Pattern::Ctor { .. }
    );

    test_expr_parser!(
        it_parses_braced_ctor_pattern_with_discard,
        "Cons { head, tail : tl, .. }",
        pattern,
        Pattern::Ctor { .. }
    );

    test_expr_parser!(
        it_parses_parenthesised_ctor_pattern,
        "Tree::Cons ( head, tl )",
        pattern,
        Pattern::Ctor { .. }
    );

    test_expr_parser!(
        it_parses_parenthesised_ctor_pattern_with_discard,
        "Tree::Leaf ( l, v, .. )",
        pattern,
        Pattern::Ctor { .. }
    );

    test_expr_parser!(
        it_parses_match_expr,
        r"match x {
            BinTree::Leaf => 0,
            BinTree::Branch(l, v, r) => {
              let res : i64 = v + 1;
              res * 5
            }
         }",
        expr,
        Expr::Match { .. }
    );

    test_expr_parser!(
        it_parses_attributed_expr,
        r"#[musttail] fib(12)",
        expr,
        Expr::Attributed { .. }
    );

    test_expr_parser!(
        it_parses_attributed_expr_with_arguments,
        r#"#[musttail(test = true, message = "123", blabla)] fib(12)"#,
        expr,
        Expr::Attributed { .. }
    );

    test_expr_parser!(
        it_parses_freeze_expr,
        r#"freeze foo(a, b, c)"#,
        expr,
        Expr::Freeze { .. }
    );

    test_expr_parser!(
        it_parses_lambda_expr,
        r#"| x : i32 | : i32 { x }"#,
        expr,
        Expr::Lambda { .. }
    );
}
