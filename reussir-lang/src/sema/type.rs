#![allow(unused)]
use super::Context;

use crate::syntax::{QualifiedName, expr::ExprPtr, r#type as syn_type};

pub enum FreezeKind {
    NonFreezing,
    Frozen,
    Unfrozen,
}

/// Semantic Type
pub enum Type<'t> {
    APInt,
    APFloat,
    Int(syn_type::Int),
    Float(syn_type::Float),
    Str,
    Bool,
    Unit,
    Never,
    Var(usize),
    Arrow(&'t Self, &'t Self),
    Forall(&'t Self),
    Exists(&'t Self),
    Record {
        name: QualifiedName<'t>,
        args: &'t [Self],
        freeze_kind: FreezeKind,
    },
}

fn infer<'t, 'e: 't>(expr: ExprPtr<'e>, ctx: &'t Context) -> Option<&'t Type<'t>> {
    todo!()
}

fn check<'t, 'e: 't>(expr: ExprPtr<'e>, ctx: &'t Context, ty: &'t Type<'t>) -> bool {
    todo!()
}

pub struct TypeBound {}

fn check_bound<'t>(ty: &'t Type<'t>, bound: TypeBound) -> bool {
    true
}
