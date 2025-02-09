use crate::syntax::r#abstract::Term;

macro_rules! both {
    ($t:pat) => {
        ($t, $t)
    };
}

#[allow(dead_code)]
pub fn convert<'src>(want: &'src Term<'src>, got: &'src Term<'src>) -> bool {
    use Term::*;
    match (want, got) {
        (Ident(a), Ident(b)) => a.id == b.id,
        both!(Type) => true,
        both!(NoneType) => true,
        both!(None) => unreachable!(),
        both!(Boolean) => true,
        both!(False) => unreachable!(),
        both!(True) => unreachable!(),
        both!(String) => true,
        both!(Str(..)) => unreachable!(),
        both!(F32) => true,
        both!(F64) => true,
        both!(Float(..)) => unreachable!(),
        (FnType { .. }, FnType { .. }) => todo!(),
        both!(Fn { .. }) => unreachable!(),
        both!(Call { .. }) => unreachable!(),
        both!(Pure) => true,
        (GenericFnType { .. }, GenericFnType { .. }) => todo!(),
        (GenericFn { .. }, GenericFn { .. }) => todo!(),

        _ => false,
    }
}
