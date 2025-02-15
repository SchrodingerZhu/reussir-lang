use std::marker::PhantomData;

use chumsky::{input::ValueInput, prelude::*};
use logos::Source;

use super::{
    Attribute, Context, FieldName, ParserExtra, Ptr, SmallCollector, WithSpan, attribute,
    expr::ExprPtr, lexer::Token, map_alloc, spanned_ident, r#type::TypePtr,
};

#[derive(Debug)]
pub enum Stmt<'ctx> {
    Attributed(Ptr<'ctx, Attribute<'ctx>>, StmtPtr<'ctx>),
    /// Global value declaration
    Val {
        name: WithSpan<&'ctx str>,
        /// TODO: should we allow inference?
        r#type: TypePtr<'ctx>,
        value: ExprPtr<'ctx>,
    },
    /// Record declaration
    Data(DataDecl<'ctx>),
    /// Function declaration
    Fn {
        proto: Ptr<'ctx, FuncProto<'ctx>>,
        body: ExprPtr<'ctx>,
    },
    /// For builtin and ffi
    ExternFn(Ptr<'ctx, FuncProto<'ctx>>),
    ExternVal {
        name: WithSpan<&'ctx str>,
        r#type: TypePtr<'ctx>,
    },
    /// TODO: we do not support multi-file yet in alpha release
    Module(WithSpan<&'ctx str>),
    Use(UseStmt),
}

/// TODO: visibility, bounds
#[derive(Debug)]
pub struct FuncProto<'ctx> {
    name: WithSpan<&'ctx str>,
    tyvars: &'ctx [WithSpan<&'ctx str>],
    freeze: bool,
    params: &'ctx [Ptr<'ctx, FuncParam<'ctx>>],
    ret: TypePtr<'ctx>,
}

#[derive(Debug)]
pub struct FuncParam<'ctx> {
    name: WithSpan<&'ctx str>,
    attrs: &'ctx [Ptr<'ctx, Attribute<'ctx>>],
    r#type: TypePtr<'ctx>,
}

#[derive(Debug)]
pub enum UseStmt {}

#[derive(Debug)]
pub struct DataDecl<'ctx> {
    name: WithSpan<&'ctx str>,
    tyvars: &'ctx [WithSpan<&'ctx str>],
    bounds: &'ctx [Ptr<'ctx, TypeBound<'ctx>>],
    ctors: &'ctx [Ptr<'ctx, CtorDecl<'ctx>>],
}

#[derive(Debug)]
pub struct CtorDecl<'ctx> {
    name: WithSpan<&'ctx str>,
    attrs: &'ctx [Ptr<'ctx, Attribute<'ctx>>],
    params: &'ctx [Ptr<'ctx, CtorParam<'ctx>>],
}

/// TODO: type bound is not supported yet
#[derive(Debug, Default)]
pub struct TypeBound<'ctx>(PhantomData<&'ctx ()>);

#[derive(Debug)]
pub struct CtorParam<'ctx> {
    name: WithSpan<FieldName<'ctx>>,
    attrs: &'ctx [Ptr<'ctx, Attribute<'ctx>>],
    r#type: TypePtr<'ctx>,
}

pub type StmtPtr<'ctx> = Ptr<'ctx, Stmt<'ctx>>;

macro_rules! stmt_parser {
    ($($vis:vis $name:ident => $body:expr)+) => {
        $($vis fn $name<'a, I, E, T>(e: E, t: T) -> impl Parser<'a, I, StmtPtr<'a>, ParserExtra<'a>> + Clone
        where
          I: ValueInput<'a, Token = Token<'a>, Span = SimpleSpan>,
          E: Parser<'a, I, ExprPtr<'a>, ParserExtra<'a>> + Clone,
          T: Parser<'a, I, TypePtr<'a>, ParserExtra<'a>> + Clone,
        {
              $body(e, t).map_with(map_alloc)
        })+
    };
}

fn annotated_stmt<'a, I, P, A, E>(
    toplevel: P,
    attr: A,
    expr: E,
) -> impl Parser<'a, I, StmtPtr<'a>, ParserExtra<'a>> + Clone
where
    I: ValueInput<'a, Token = Token<'a>, Span = SimpleSpan>,
    A: Parser<'a, I, Ptr<'a, Attribute<'a>>, ParserExtra<'a>> + Clone,
    P: Parser<'a, I, StmtPtr<'a>, ParserExtra<'a>> + Clone,
    E: Parser<'a, I, ExprPtr<'a>, ParserExtra<'a>> + Clone,
{
    attr.then(toplevel)
        .map(|(e, t)| Stmt::Attributed(e, t))
        .map_with(map_alloc)
}

fn named_ctor_params<'a, I, A, T>(
    attr: A,
    r#type: T,
) -> impl Parser<'a, I, &'a [Ptr<'a, CtorParam<'a>>], ParserExtra<'a>> + Clone
where
    I: ValueInput<'a, Token = Token<'a>, Span = SimpleSpan>,
    A: Parser<'a, I, Ptr<'a, Attribute<'a>>, ParserExtra<'a>> + Clone,
    T: Parser<'a, I, TypePtr<'a>, ParserExtra<'a>> + Clone,
{
    let attrs = attr
        .repeated()
        .collect::<SmallCollector<_, 2>>()
        .map_with(|c, m| {
            let state: &&Context = m.state();
            state.alloc_slice(c.0)
        });
    attrs
        .then(spanned_ident().map(|x| WithSpan(FieldName::Name(x.0), x.1)))
        .then_ignore(just(Token::Colon))
        .then(r#type)
        .map(|((attrs, name), r#type)| CtorParam {
            attrs,
            name,
            r#type,
        })
        .map_with(map_alloc)
        .separated_by(just(Token::Comma))
        .collect::<SmallCollector<_, 4>>()
        .map_with(|c, m| {
            let state: &&Context = m.state();
            state.alloc_slice(c.0)
        })
}

fn unamed_ctor_params<'a, I, A, T>(
    attr: A,
    r#type: T,
) -> impl Parser<'a, I, &'a [Ptr<'a, CtorParam<'a>>], ParserExtra<'a>> + Clone
where
    I: ValueInput<'a, Token = Token<'a>, Span = SimpleSpan>,
    A: Parser<'a, I, Ptr<'a, Attribute<'a>>, ParserExtra<'a>> + Clone,
    T: Parser<'a, I, TypePtr<'a>, ParserExtra<'a>> + Clone,
{
    let attrs = attr
        .repeated()
        .collect::<SmallCollector<_, 2>>()
        .map_with(|c, m| {
            let state: &&Context = m.state();
            state.alloc_slice(c.0)
        });
    attrs
        .then(r#type.map_with(|x, m| (x, m.span())))
        .map_with(|(attrs, r#type), m| (attrs, r#type, m.span()))
        .separated_by(just(Token::Comma))
        .collect::<SmallCollector<_, 4>>()
        .map_with(|c, m| {
            let state: &&Context = m.state();
            state.alloc_slice(c.0.into_iter().enumerate().map(
                |(idx, (attrs, (r#type, type_span), span))| {
                    state.alloc(WithSpan(
                        CtorParam {
                            attrs,
                            r#type,
                            name: WithSpan(FieldName::Idx(idx), type_span),
                        },
                        span,
                    ))
                },
            ))
        })
}

fn ctor_decl<'a, I, A, T>(
    attr: A,
    r#type: T,
) -> impl Parser<'a, I, Ptr<'a, CtorDecl<'a>>, ParserExtra<'a>> + Clone
where
    I: ValueInput<'a, Token = Token<'a>, Span = SimpleSpan>,
    A: Parser<'a, I, Ptr<'a, Attribute<'a>>, ParserExtra<'a>> + Clone,
    T: Parser<'a, I, TypePtr<'a>, ParserExtra<'a>> + Clone,
{
    let named_params = named_ctor_params(attr.clone(), r#type.clone())
        .delimited_by(just(Token::LBrace), just(Token::RBrace));
    let unamed_params = unamed_ctor_params(attr.clone(), r#type)
        .delimited_by(just(Token::LParen), just(Token::RParen));
    let attrs = attr
        .repeated()
        .collect::<SmallCollector<_, 2>>()
        .map_with(|c, m| {
            let state: &&Context = m.state();
            state.alloc_slice(c.0)
        });
    let params = named_params
        .or(unamed_params)
        .repeated()
        .at_most(1)
        .collect::<SmallCollector<_, 1>>()
        .map(|c| c.0.into_iter().next().unwrap_or(&[]));
    attrs
        .then(spanned_ident())
        .then(params)
        .map(|((attrs, name), params)| CtorDecl {
            attrs,
            name,
            params,
        })
        .map_with(map_alloc)
}

fn type_vars<'a, I>() -> impl Parser<'a, I, &'a [WithSpan<&'a str>], ParserExtra<'a>> + Clone
where
    I: ValueInput<'a, Token = Token<'a>, Span = SimpleSpan>,
{
    spanned_ident()
        .separated_by(just(Token::Comma))
        .collect::<SmallCollector<_, 4>>()
        .map_with(|c, m| {
            let state: &&Context = m.state();
            state.alloc_slice(c.0)
        })
        .delimited_by(just(Token::LSquare), just(Token::RSquare))
        .repeated()
        .at_most(1)
        .collect::<SmallCollector<_, 1>>()
        .map(|c| c.0.into_iter().next().unwrap_or(&[]))
}

fn data_decl<'a, I, A, T>(
    attr: A,
    r#type: T,
) -> impl Parser<'a, I, StmtPtr<'a>, ParserExtra<'a>> + Clone
where
    I: ValueInput<'a, Token = Token<'a>, Span = SimpleSpan>,
    A: Parser<'a, I, Ptr<'a, Attribute<'a>>, ParserExtra<'a>> + Clone,
    T: Parser<'a, I, TypePtr<'a>, ParserExtra<'a>> + Clone,
{
    let tyvars = type_vars();
    let ctors = ctor_decl(attr, r#type)
        .separated_by(just(Token::Or))
        .at_least(1)
        .collect::<SmallCollector<_, 4>>()
        .map_with(|c, m| {
            let state: &&Context = m.state();
            state.alloc_slice(c.0)
        });
    just(Token::Data)
        .ignore_then(spanned_ident())
        .then(tyvars)
        .then_ignore(just(Token::Eq))
        .then(ctors)
        .map(|((name, tyvars), ctors)| {
            Stmt::Data(DataDecl {
                name,
                tyvars,
                ctors,
                bounds: &[],
            })
        })
        .map_with(map_alloc)
}

fn func_params<'a, I, A, T>(
    attr: A,
    r#type: T,
) -> impl Parser<'a, I, &'a [Ptr<'a, FuncParam<'a>>], ParserExtra<'a>> + Clone
where
    I: ValueInput<'a, Token = Token<'a>, Span = SimpleSpan>,
    A: Parser<'a, I, Ptr<'a, Attribute<'a>>, ParserExtra<'a>> + Clone,
    T: Parser<'a, I, TypePtr<'a>, ParserExtra<'a>> + Clone,
{
    let attrs = attr
        .repeated()
        .collect::<SmallCollector<_, 2>>()
        .map_with(|c, m| {
            let state: &&Context = m.state();
            state.alloc_slice(c.0)
        });
    attrs
        .then(spanned_ident())
        .then_ignore(just(Token::Colon))
        .then(r#type)
        .map(|((attrs, name), r#type)| FuncParam {
            attrs,
            name,
            r#type,
        })
        .map_with(map_alloc)
        .separated_by(just(Token::Comma))
        .collect::<SmallCollector<_, 4>>()
        .map_with(|c, m| {
            let state: &&Context = m.state();
            state.alloc_slice(c.0)
        })
        .delimited_by(just(Token::LParen), just(Token::RParen))
        .or(just(Token::Unit).to(&[] as &'a [Ptr<'a, FuncParam<'a>>]))
}

fn func_proto<'a, I, A, T>(
    attr: A,
    r#type: T,
) -> impl Parser<'a, I, Ptr<'a, FuncProto<'a>>, ParserExtra<'a>> + Clone
where
    I: ValueInput<'a, Token = Token<'a>, Span = SimpleSpan>,
    A: Parser<'a, I, Ptr<'a, Attribute<'a>>, ParserExtra<'a>> + Clone,
    T: Parser<'a, I, TypePtr<'a>, ParserExtra<'a>> + Clone,
{
    let freeze = just(Token::Freeze)
        .repeated()
        .at_most(1)
        .count()
        .map(|x| x != 0);
    freeze
        .then_ignore(just(Token::Fn))
        .then(spanned_ident())
        .then(type_vars())
        .then(func_params(attr, r#type.clone()))
        .then_ignore(just(Token::Colon))
        .then(r#type)
        .map(|((((freeze, name), tyvars), params), ret)| FuncProto {
            freeze,
            name,
            params,
            ret,
            tyvars,
        })
        .map_with(map_alloc)
}

fn extern_fn_stmt<'a, I, A, T>(
    attr: A,
    r#type: T,
) -> impl Parser<'a, I, StmtPtr<'a>, ParserExtra<'a>> + Clone
where
    I: ValueInput<'a, Token = Token<'a>, Span = SimpleSpan>,
    A: Parser<'a, I, Ptr<'a, Attribute<'a>>, ParserExtra<'a>> + Clone,
    T: Parser<'a, I, TypePtr<'a>, ParserExtra<'a>> + Clone,
{
    just(Token::Extern)
        .ignore_then(func_proto(attr, r#type))
        .map(Stmt::ExternFn)
        .map_with(map_alloc)
}

fn fn_stmt<'a, I, A, E, T>(
    attr: A,
    expr: E,
    r#type: T,
) -> impl Parser<'a, I, StmtPtr<'a>, ParserExtra<'a>> + Clone
where
    I: ValueInput<'a, Token = Token<'a>, Span = SimpleSpan>,
    A: Parser<'a, I, Ptr<'a, Attribute<'a>>, ParserExtra<'a>> + Clone,
    T: Parser<'a, I, TypePtr<'a>, ParserExtra<'a>> + Clone,
    E: Parser<'a, I, ExprPtr<'a>, ParserExtra<'a>> + Clone,
{
    func_proto(attr, r#type)
        .then_ignore(just(Token::Eq))
        .then(expr)
        .map(|(proto, body)| Stmt::Fn { proto, body })
        .map_with(map_alloc)
}

pub fn stmt<'a, I>() -> impl Parser<'a, I, StmtPtr<'a>, ParserExtra<'a>> + Clone
where
    I: ValueInput<'a, Token = Token<'a>, Span = SimpleSpan>,
{
    let expr = super::expr::expr::<I>();
    let r#type = super::r#type::r#type::<I>();
    let attr = super::attribute(expr.clone());
    recursive(move |stmt| {
        choice((
            annotated_stmt(stmt.clone(), attr.clone(), expr.clone()),
            val_stmt(expr.clone(), r#type.clone()),
            extern_fn_stmt(attr.clone(), r#type.clone()),
            extern_val_stmt(expr.clone(), r#type.clone()),
            fn_stmt(attr.clone(), expr, r#type.clone()),
            data_decl(attr, r#type.clone()),
        ))
    })
}

stmt_parser! {

    val_stmt => | expr : E, r#type : T | {
        just(Token::Let)
            .ignore_then(spanned_ident())
            .then_ignore(just(Token::Colon))
            .then(r#type)
            .then_ignore(just(Token::Eq))
            .then(expr)
            .map(|((name, r#type), value)| Stmt::Val {name, r#type, value})
    }

    extern_val_stmt => | _, r#type : T | {
        just(Token::Extern)
            .ignore_then(spanned_ident())
            .then_ignore(just(Token::Colon))
            .then(r#type)
            .map(|(name, r#type)| Stmt::ExternVal { name, r#type })
    }
}

#[cfg(test)]
mod test {
    use super::*;
    macro_rules! test_stmt_parser {
        ($name:ident, $src:literal, $pattern:pat) => {
            #[test]
            fn $name() {
                let ctx = Context::from_src($src);
                let stream = ctx.token_stream();
                let res = stmt().parse_with_state(stream, &mut &ctx).unwrap();
                println!("{:#?}", res);
                assert!(matches!(res.0, $pattern));
            }
        };
    }

    test_stmt_parser!(
        it_parses_val_stmt,
        r#"let hello_world : str = "hello, world!\n""#,
        Stmt::Val { .. }
    );

    test_stmt_parser!(
        it_parses_extern_val_stmt,
        r#"extern hello_world : str"#,
        Stmt::ExternVal { .. }
    );

    test_stmt_parser!(
        it_parses_single_no_param_data_stmt,
        r#"data Test = Test"#,
        Stmt::Data { .. }
    );

    test_stmt_parser!(
        it_parses_single_unamed_data_stmt,
        r#"data Test[A, B, C] = Test(A, B, #[test] C)"#,
        Stmt::Data { .. }
    );

    test_stmt_parser!(
        it_parses_single_named_data_stmt,
        r#"data Test[A, B, C] = Test{a : A, b : B, c : C}"#,
        Stmt::Data { .. }
    );

    test_stmt_parser!(
        it_parses_sum_data_stmt,
        r#"data Test[A] = #[default] Foo
                        | Bar(i32, i32)
                        | Baz { a : A, #[regional] b : i32, c : bf16 }"#,
        Stmt::Data { .. }
    );

    test_stmt_parser!(
        it_parses_extern_fn_stmt_with_zero_params,
        r#"extern fn foo() : i32"#,
        Stmt::ExternFn { .. }
    );

    test_stmt_parser!(
        it_parses_extern_fn_stmt,
        r#"extern fn foo(a : i32, b : i32, #[check_nonzero] c : i32) : i32"#,
        Stmt::ExternFn { .. }
    );

    test_stmt_parser!(
        it_parses_fn_stmt,
        r#"fn foo(a : i32, b : i32, c : i32) : i32 = a + b + c"#,
        Stmt::Fn { .. }
    );
}
