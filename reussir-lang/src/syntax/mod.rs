#![allow(unused)]
use std::{cell::RefCell, iter::Inspect, ops::Deref, path::Path};

use bumpalo::Bump;
use chumsky::{
    container::Container,
    error::Rich,
    extra::{Full, SimpleState},
    input::{Checkpoint, Cursor, Input, MapExtra, Stream, ValueInput},
    inspector::Inspector,
    span::SimpleSpan,
    ParseResult, Parser,
};
use expr::ExprPtr;
use lexer::Token;
use logos::Logos;
use rustc_hash::FxHashMapRand;
use smallvec::SmallVec;
use stmt::StmtPtr;
use thiserror::Error;

pub mod expr;
mod lexer;
pub mod stmt;
pub mod r#type;

struct SmallCollector<T, const N: usize>(SmallVec<T, N>);

impl<T, const N: usize> Default for SmallCollector<T, N> {
    fn default() -> Self {
        Self(Default::default())
    }
}

impl<T, const N: usize> Container<T> for SmallCollector<T, N> {
    fn with_capacity(n: usize) -> Self {
        Self(SmallVec::with_capacity(n))
    }

    fn push(&mut self, item: T) {
        self.0.push(item);
    }
}

#[derive(Debug, Copy, Clone)]
pub struct WithSpan<T>(pub T, pub SimpleSpan);

unsafe impl<'a, T: gc_arena::Collect<'a>> gc_arena::Collect<'a> for WithSpan<T> {
    #[inline]
    fn trace<C: gc_arena::collect::Trace<'a>>(&self, cc: &mut C) {
        self.0.trace(cc);
    }
}

type Ptr<'ctx, T> = &'ctx WithSpan<T>;

impl<T: PartialEq> PartialEq for WithSpan<T> {
    fn eq(&self, other: &Self) -> bool {
        self.0 == other.0
    }
}

impl<T: Eq> Eq for WithSpan<T> {}

impl<T> std::hash::Hash for WithSpan<T>
where
    T: std::hash::Hash,
{
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.0.hash(state);
    }
}

impl<T> Deref for WithSpan<T> {
    type Target = T;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

pub struct Context {
    arena: Bump,
    input: Option<std::path::PathBuf>,
    src: String,
}

#[derive(Debug, Copy, Clone, Eq, PartialEq, Hash)]
pub struct QualifiedName<'ctx>(&'ctx [&'ctx str], &'ctx str);

impl<'a> QualifiedName<'a> {
    pub fn new(qualifier: &'a [&'a str], basename: &'a str) -> Self {
        Self(qualifier, basename)
    }
    pub fn qualifier(&self) -> &[&str] {
        self.0
    }
    pub fn is_unqualified(&self) -> bool {
        self.0.is_empty()
    }
    pub fn basename(&self) -> &str {
        self.1
    }
}

#[derive(Error, Debug)]
pub enum Error {
    #[error("failed to read source from input {0}")]
    IoError(#[from] std::io::Error),
}

type Result<T> = std::result::Result<T, Error>;

impl Context {
    pub fn from_file<P: AsRef<Path>>(input: P) -> Result<Self> {
        let input = input.as_ref().to_owned();
        let src = std::fs::read_to_string(&input)?;
        let input = Some(input);
        Ok(Self::new(input, src))
    }

    pub fn from_src<S: AsRef<str>>(src: S) -> Self {
        Self::new(None, src.as_ref().to_string())
    }

    pub fn alloc<T>(&self, data: T) -> &T {
        self.arena.alloc(data)
    }

    pub fn alloc_str<S: AsRef<str>>(&self, data: S) -> &str {
        self.arena.alloc_str(data.as_ref())
    }

    pub fn alloc_slice<T, I: IntoIterator<Item = T>>(&self, data: I) -> &[T]
    where
        I::IntoIter: ExactSizeIterator,
    {
        self.arena.alloc_slice_fill_iter(data)
    }

    fn new(input: Option<std::path::PathBuf>, src: String) -> Self {
        Self {
            arena: Bump::new(),
            input,
            src,
        }
    }
    fn new_qualified_name<'a, I>(&'a self, qualifiers: I, basename: &'a str) -> QualifiedName<'a>
    where
        I: IntoIterator<Item = &'a str>,
        I::IntoIter: ExactSizeIterator,
    {
        let qualifiers = self.arena.alloc_slice_fill_iter(qualifiers);
        QualifiedName(qualifiers, basename)
    }

    pub fn parse(&self) -> ParseResult<&[StmtPtr], Rich<Token>> {
        use chumsky::prelude::*;
        let mut this: &Self = self;
        stmt::stmt()
            .separated_by(just(Token::Semi))
            .allow_trailing()
            .collect::<Vec<_>>()
            .map(|x| self.alloc_slice(x))
            .parse_with_state(self.token_stream(), &mut this)
    }
}

type ParserExtra<'a> = chumsky::extra::Full<Rich<'a, lexer::Token<'a>>, &'a Context, ()>;

impl<'src, I: Input<'src>> Inspector<'src, I> for &'src Context {
    type Checkpoint = ();
    #[inline(always)]
    fn on_token(&mut self, _: &<I as Input<'src>>::Token) {}
    #[inline(always)]
    fn on_save<'parse>(&self, _: &Cursor<'src, 'parse, I>) -> Self::Checkpoint {}
    #[inline(always)]
    fn on_rewind<'parse>(&mut self, _: &Checkpoint<'src, 'parse, I, Self::Checkpoint>) {}
}

fn map_alloc<'src, I, T>(
    value: T,
    map: &mut MapExtra<'src, '_, I, ParserExtra<'src>>,
) -> &'src WithSpan<T>
where
    I: Input<'src, Token = lexer::Token<'src>, Span = SimpleSpan>,
{
    let span = map.span();
    map.state().alloc(WithSpan(value, span))
}

fn qualified_name<'a, I>(
) -> impl Parser<'a, I, &'a WithSpan<QualifiedName<'a>>, ParserExtra<'a>> + Clone
where
    I: ValueInput<'a, Token = Token<'a>, Span = SimpleSpan>,
{
    use chumsky::prelude::*;
    let ident = select! {
        Token::Ident(x) => x
    };
    let prefix = ident
        .then_ignore(just(Token::PathSep))
        .repeated()
        .collect::<SmallCollector<_, 4>>();

    prefix.then(ident).map_with(|(prefix, basename), m| {
        let state: &mut &'a Context = m.state();
        let prefix = state.alloc_slice(prefix.0);
        let name = QualifiedName(prefix, basename);
        map_alloc(name, m)
    })
}

#[derive(Debug, Clone)]
pub struct Attribute<'ctx> {
    name: WithSpan<&'ctx str>,
    arguments: &'ctx [Ptr<'ctx, AttributeArgument<'ctx>>],
}

#[derive(Debug, Clone)]
pub enum AttributeArgument<'ctx> {
    KwArg(WithSpan<&'ctx str>, ExprPtr<'ctx>),
    Expr(ExprPtr<'ctx>),
}

fn attribute_argument<'a, I, P>(
    expr: P,
) -> impl Parser<'a, I, &'a WithSpan<AttributeArgument<'a>>, ParserExtra<'a>> + Clone
where
    I: ValueInput<'a, Token = Token<'a>, Span = SimpleSpan>,
    P: Parser<'a, I, ExprPtr<'a>, ParserExtra<'a>> + Clone,
{
    use chumsky::prelude::*;
    let ident = select! {
        Token::Ident(x) = m => WithSpan(x, m.span())
    };
    let expr_argument = expr.clone().map(AttributeArgument::Expr);
    let keyword_argument = ident
        .then_ignore(just(Token::Eq))
        .then(expr)
        .map(|(k, e)| AttributeArgument::KwArg(k, e));
    keyword_argument.or(expr_argument).map_with(map_alloc)
}

// expr is needed because this can be used also in expression where the expr parser is using the recursive reference.
fn attribute<'a, I, P>(
    expr: P,
) -> impl Parser<'a, I, &'a WithSpan<Attribute<'a>>, ParserExtra<'a>> + Clone
where
    I: ValueInput<'a, Token = Token<'a>, Span = SimpleSpan>,
    P: Parser<'a, I, ExprPtr<'a>, ParserExtra<'a>> + Clone,
{
    use chumsky::prelude::*;
    let ident = select! {
        Token::Ident(x) = m => WithSpan(x, m.span())
    };
    let arguments = attribute_argument(expr)
        .separated_by(just(Token::Comma))
        .at_least(1)
        .collect::<SmallCollector<_, 4>>()
        .map_with(|c, m| {
            let state: &&Context = m.state();
            state.alloc_slice(c.0)
        })
        .delimited_by(just(Token::LParen), just(Token::RParen))
        .repeated()
        .at_most(1)
        .collect::<SmallCollector<_, 1>>()
        .map(|x| x.0.into_iter().next().unwrap_or(&[]));
    just(Token::Sharp).ignore_then(
        ident
            .then(arguments)
            .map(|(name, arguments)| Attribute { name, arguments })
            .map_with(map_alloc)
            .delimited_by(just(Token::LSquare), just(Token::RSquare)),
    )
}

#[derive(Debug, Clone, Copy, Eq, PartialEq, Hash)]
pub enum FieldName<'ctx> {
    Idx(usize),
    Name(&'ctx str),
}

fn spanned_ident<'a, I>() -> impl Parser<'a, I, WithSpan<&'a str>, ParserExtra<'a>> + Copy
where
    I: ValueInput<'a, Token = Token<'a>, Span = SimpleSpan>,
{
    chumsky::select! { Token::Ident(x) = m => WithSpan(x, m.span()) }
}

#[cfg(test)]
mod test {
    use super::*;
    #[test]
    fn is_parses_example_program() {
        let src = r#"
data RbTree = Leaf | Branch(RbTree, i32, RbTree);

#[plain] // not boxed by RC
data Option[T] = None | Some(T);

fn value_add_one(t : RbTree) : Option[i32] =
   match t {
     RbTree::Leaf => None,
     RbTree::Branch(l, v, r) => Some(v + 1)
   };


"#;
        let context = Context::from_src(src);
        let ast = context.parse();
        assert!(!ast.has_errors() && ast.has_output());
        print!("{:#?}", ast.unwrap());
    }

    #[test]
    fn is_parses_map_program() {
        let src = r#"
data RbTree[T] = Leaf | Branch(RbTree[T], i32, RbTree[T]);

fn map[T, U](x : RbTree[T], f : fn(T) -> U) : RbTree[U] =
   match t {
     RbTree::Leaf => RbTree::Leaf,
     RbTree::Branch(l, v, r) => {
       let l = map(l, f);
       let v = f(v);
       let r = map(r, f);
       RbTree::Branch(l, v, r)
     }
   };
"#;
        let context = Context::from_src(src);
        let ast = context.parse();
        assert!(!ast.has_errors() && ast.has_output());
        print!("{:#?}", ast.unwrap());
    }

    #[test]
    fn is_parses_lambda_program() {
        let src = r#"
fn curry[X, Y, Z]( f : fn (X, Y) -> Z ) : fn (X) -> fn(Y) -> Z =
   | x | | y | f (x, y)
"#;
        let context = Context::from_src(src);
        let ast = context.parse();
        assert!(!ast.has_errors() && ast.has_output());
        print!("{:#?}", ast.unwrap());
    }
}
