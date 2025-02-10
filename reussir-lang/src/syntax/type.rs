#![allow(unused)]

use super::QualifiedName;
use chumsky::{container::Seq, span::SimpleSpan};
use std::collections::hash_map::Entry;

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum Type<'ctx> {
    Int(Int),
    Float(Float),
    Str,
    SrcLoc(TypePtr<'ctx>, SimpleSpan),
    #[allow(clippy::enum_variant_names)]
    TypeName(QualifiedName<'ctx>),
}

type TypePtr<'ctx> = &'ctx Type<'ctx>;

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
    name: QualifiedName<'ctx>,
    ctors: &'ctx [Ctor<'ctx>],
    is_rc_managed: bool,
}

macro_rules! impl_integer_getter {
    ($name:ident, $width:expr, $signed:expr) => {
        pub fn $name(&self) -> TypePtr {
            const TYPE: Type = Type::Int(Int::new($width, $signed));
            &TYPE
        }
    };
}

macro_rules! impl_float_getter {
    ($name:ident, $variant:ident) => {
        pub fn $name(&self) -> TypePtr {
            const TYPE: Type = Type::Float(Float::$variant);
            &TYPE
        }
    };
}

impl<'ctx> super::Context<'ctx> {
    impl_integer_getter!(get_i8_type, 8, true);
    impl_integer_getter!(get_u8_type, 8, false);
    impl_integer_getter!(get_i16_type, 16, true);
    impl_integer_getter!(get_u16_type, 16, false);
    impl_integer_getter!(get_i32_type, 32, true);
    impl_integer_getter!(get_u32_type, 32, false);
    impl_integer_getter!(get_i64_type, 64, true);
    impl_integer_getter!(get_u64_type, 64, false);
    impl_integer_getter!(get_i128_type, 128, true);
    impl_integer_getter!(get_u128_type, 128, false);
    impl_float_getter!(get_f16_type, F16);
    impl_float_getter!(get_bf16_type, BF16);
    impl_float_getter!(get_f32_type, F32);
    impl_float_getter!(get_f64_type, F64);
    impl_float_getter!(get_f128_type, F128);
    pub fn get_str_type(&self) -> TypePtr {
        const TYPE: Type = Type::Str;
        &TYPE
    }
    pub fn new_record<I>(
        &'ctx mut self,
        name: QualifiedName<'ctx>,
        ctors: I,
        is_rc_managed: bool,
    ) -> bool
    where
        I: IntoIterator<Item = Ctor<'ctx>>,
        I::IntoIter: ExactSizeIterator,
    {
        match self.records.entry(name) {
            Entry::Occupied(occupied) => false,
            Entry::Vacant(vacant) => {
                let ctors = self.arena.alloc_slice_fill_iter(ctors);
                let record = Record {
                    name,
                    ctors,
                    is_rc_managed,
                };
                vacant.insert(record);
                true
            }
        }
    }
    pub fn lookup_record(&self, name: QualifiedName<'ctx>) -> Option<&Record> {
        self.records.get(&name)
    }
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn int_correctly_records_width() {
        assert_eq!(Int::new(16, true).width(), 16);
    }
}
