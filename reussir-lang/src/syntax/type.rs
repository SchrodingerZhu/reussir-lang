#![allow(unused)]

use super::Context;
use super::lexer::Token;
use super::{ParserExtra, SmallCollector};
use super::{QualifiedName, WithSpan, map_alloc, qualified_name};
use chumsky::prelude::*;
use chumsky::{container::Seq, input::ValueInput, span::SimpleSpan};
use std::collections::hash_map::Entry;

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum Type<'ctx> {
    Int(Int),
    Float(Float),
    Char,
    Str,
    Boolean,
    Unit,
    Never,
    Func {
        args: &'ctx [TypePtr<'ctx>],
        ret: TypePtr<'ctx>,
    },
    #[allow(clippy::enum_variant_names)]
    TypeExpr {
        name: &'ctx WithSpan<QualifiedName<'ctx>>,
        args: &'ctx [TypePtr<'ctx>],
        modifier: Modifier,
    },
    //TODO: type variable
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Modifier {
    None,
    Unfrozen,
    Frozen,
}

type TypePtr<'ctx> = &'ctx WithSpan<Type<'ctx>>;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Int {
    width_ilog: u32,
    signed: bool,
}

impl Int {
    pub const fn new(width: usize, signed: bool) -> Self {
        Self {
            width_ilog: width.ilog2(),
            signed,
        }
    }
    pub fn width(&self) -> usize {
        1 << self.width_ilog
    }
    pub fn is_signed(&self) -> bool {
        self.signed
    }
}

#[derive(Debug, Clone, Copy, Eq, PartialEq, Hash)]
pub enum Float {
    F16,
    BF16,
    F32,
    F64,
    F128,
}
#[derive(Debug, Clone, Copy, Eq, PartialEq, Hash)]
pub enum FieldName<'ctx> {
    Idx(usize),
    Name(&'ctx str),
}
#[derive(Debug, Clone, Copy, Eq, PartialEq, Hash)]
pub enum FieldKind {
    Plain,
    // TODO: should also track atomicity
    LocallyMutable { is_frozen: bool },
}
#[derive(Debug, Clone, Eq, PartialEq, Hash)]
pub struct Field<'ctx> {
    name: FieldName<'ctx>,
    kind: FieldKind,
    r#type: TypePtr<'ctx>,
}

#[derive(Debug, Clone, Eq, PartialEq, Hash)]
pub struct Ctor<'ctx> {
    name: &'ctx str,
    fields: &'ctx [Field<'ctx>],
}

#[derive(Debug, Clone, Eq, PartialEq, Hash)]
pub struct Record<'ctx> {
    name: WithSpan<QualifiedName<'ctx>>,
    ctors: &'ctx [Ctor<'ctx>],
    is_rc_managed: bool,
}

impl<'ctx> super::Context<'ctx> {
    pub fn new_record_in_context<I>(
        &'ctx self,
        basename: WithSpan<&'ctx str>,
        ctors: I,
        is_rc_managed: bool,
        location: SimpleSpan,
    ) -> bool
    where
        I: IntoIterator<Item = Ctor<'ctx>>,
        I::IntoIter: ExactSizeIterator,
    {
        let name = self.new_qualified_name(self.scope.iter().copied(), basename.0);
        let name = WithSpan(name, basename.1);
        match self.records.borrow_mut().entry(name.0) {
            Entry::Occupied(occupied) => false,
            Entry::Vacant(vacant) => {
                let ctors = self.arena.alloc_slice_fill_iter(ctors);
                let record = Record {
                    name,
                    ctors,
                    is_rc_managed,
                };
                vacant.insert(WithSpan(self.arena.alloc(record), location));
                true
            }
        }
    }
    pub fn new_record<I>(
        &'ctx self,
        name: WithSpan<QualifiedName<'ctx>>,
        ctors: I,
        is_rc_managed: bool,
        location: SimpleSpan,
    ) -> bool
    where
        I: IntoIterator<Item = Ctor<'ctx>>,
        I::IntoIter: ExactSizeIterator,
    {
        match self.records.borrow_mut().entry(name.0) {
            Entry::Occupied(occupied) => false,
            Entry::Vacant(vacant) => {
                let ctors = self.arena.alloc_slice_fill_iter(ctors);
                let record = Record {
                    name,
                    ctors,
                    is_rc_managed,
                };
                vacant.insert(WithSpan(self.arena.alloc(record), location));
                true
            }
        }
    }
    pub fn lookup_record(&self, name: QualifiedName) -> Option<WithSpan<&Record>> {
        self.records.borrow().get(&name).copied()
    }
}

fn func_type<'a, I, P>(toplevel: P) -> impl Parser<'a, I, TypePtr<'a>, ParserExtra<'a>> + Clone
where
    I: ValueInput<'a, Token = Token<'a>, Span = SimpleSpan>,
    P: Parser<'a, I, TypePtr<'a>, ParserExtra<'a>> + Clone,
{
    let arglist = toplevel
        .clone()
        .separated_by(just(Token::Comma))
        .collect::<SmallCollector<_, 4>>()
        .delimited_by(just(Token::LParen), just(Token::RParen))
        .or(just(Token::Unit)
            .ignored()
            .map(|_| SmallCollector::default()));
    just(Token::Fn)
        .ignore_then(arglist)
        .then_ignore(just(Token::RArrow))
        .then(toplevel)
        .map_with(|(list, ret), m| {
            let state: &'a Context<'a> = m.state();
            let args = state.alloc_slice(list.0);
            let func = Type::Func { args, ret };
            map_alloc(func, m)
        })
        .labelled("function type")
}

fn type_expr<'a, I, P>(toplevel: P) -> impl Parser<'a, I, TypePtr<'a>, ParserExtra<'a>> + Clone
where
    I: ValueInput<'a, Token = Token<'a>, Span = SimpleSpan>,
    P: Parser<'a, I, TypePtr<'a>, ParserExtra<'a>> + Clone,
{
    let modifier = just(Token::At)
        .to(Modifier::Unfrozen)
        .or(just(Token::Star).to(Modifier::Frozen))
        .repeated()
        .at_most(1)
        .collect::<SmallCollector<_, 1>>()
        .map(|x| x.0.first().copied().unwrap_or(Modifier::None));

    let application = toplevel
        .separated_by(just(Token::Comma))
        .collect::<SmallCollector<_, 4>>()
        .delimited_by(just(Token::LSquare), just(Token::RSquare));

    modifier
        .then(qualified_name())
        .then(application)
        .map_with(|((modifier, name), app), m| {
            let state: &'a Context<'a> = m.state();
            let args = state.alloc_slice(app.0);
            let expr = Type::TypeExpr {
                name,
                args,
                modifier,
            };
            map_alloc(expr, m)
        })
        .labelled("type expression")
}

macro_rules! type_parser {
    ($($name:ident => $body:block)+) => {
        $(fn $name<'a, I>() -> impl Parser<'a, I, TypePtr<'a>, ParserExtra<'a>> + Clone
        where
            I: ValueInput<'a, Token = Token<'a>, Span = SimpleSpan>,
        $body)+
    };
}

type_parser! {

    integer_type => {
        select! {
            Token::I8 => Type::Int(Int::new(8, true)),
            Token::I16 => Type::Int(Int::new(16, true)),
            Token::I32 => Type::Int(Int::new(32, true)),
            Token::I64 => Type::Int(Int::new(64, true)),
            Token::I128 => Type::Int(Int::new(128, true)),
            Token::U8 => Type::Int(Int::new(8, false)),
            Token::U16 => Type::Int(Int::new(16, false)),
            Token::U32 => Type::Int(Int::new(32, false)),
            Token::U64 => Type::Int(Int::new(64, false)),
            Token::U128 => Type::Int(Int::new(128, false)),
        }
        .map_with(map_alloc)
        .labelled("integer type")
    }

    float_type => {
        select! {
            Token::F16 => Type::Float(Float::F16),
            Token::BF16 => Type::Float(Float::BF16),
            Token::F32 => Type::Float(Float::F32),
            Token::F64 => Type::Float(Float::F64),
            Token::F128 => Type::Float(Float::F128),
        }
        .map_with(map_alloc)
        .labelled("floating-point type")
    }

    char_type => {
        just(Token::Char).to(Type::Char).map_with(map_alloc).labelled("character type")
    }

    str_type => {
        just(Token::Str).to(Type::Str).map_with(map_alloc).labelled("str type")
    }

    boolean_type => {
        just(Token::Bool).to(Type::Boolean).map_with(map_alloc).labelled("boolean type")
    }

    unit_type => {
        just(Token::Unit).to(Type::Unit).map_with(map_alloc).labelled("unit type")
    }

    bottom_type => {
        just(Token::Not).to(Type::Never).map_with(map_alloc).labelled("never type")
    }

    r#type => {
        recursive( |toplevel| {
            let func = func_type(toplevel.clone());
            let expr = type_expr(toplevel);
            choice((
                integer_type(),
                float_type(),
                char_type(),
                str_type(),
                boolean_type(),
                unit_type(),
                bottom_type(),
                func,
                expr,
            ))
        })
    }
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn int_correctly_records_width() {
        assert_eq!(Int::new(16, true).width(), 16);
    }

    #[test]
    fn it_parses_unit_type() {
        let ctx = Context::from_src("()");
        let stream = ctx.token_stream();
        let res = r#type().parse_with_state(stream, &mut &ctx).unwrap();
        assert!(matches!(res.0, Type::Unit));
        println!("{:#?}", res)
    }

    #[test]
    fn it_parses_never_type() {
        let ctx = Context::from_src("!");
        let stream = ctx.token_stream();
        let res = r#type().parse_with_state(stream, &mut &ctx).unwrap();
        assert!(matches!(res.0, Type::Never));
        println!("{:#?}", res)
    }

    #[test]
    fn it_parses_type_expr() {
        let ctx = Context::from_src(
            "Vec[i32, @std::Vec[i32], (), bool, f128, std::Vec[*std::Vec[std::ds::foo[i32, i32, i32]]]]",
        );
        let stream = ctx.token_stream();
        let res = r#type().parse_with_state(stream, &mut &ctx).unwrap();
        assert!(matches!(res.0, Type::TypeExpr { .. }));
        println!("{:#?}", res)
    }
    #[test]
    fn it_parses_func_type() {
        let ctx = Context::from_src("fn (Vec[i32], (), ()) -> fn ((), ()) -> Result[bool, str]");
        let stream = ctx.token_stream();
        let res = r#type().parse_with_state(stream, &mut &ctx).unwrap();
        assert!(matches!(res.0, Type::Func { .. }));
        println!("{:#?}", res)
    }
    #[test]
    fn it_parses_special_func_type() {
        let ctx = Context::from_src("fn () -> fn (()) -> !");
        let stream = ctx.token_stream();
        let res = r#type().parse_with_state(stream, &mut &ctx).unwrap();
        assert!(matches!(res.0, Type::Func { .. }));
        println!("{:#?}", res)
    }
}
