use std::rc::Rc;

use either::Either;

use crate::utils::{Icit, Name, Pruning, WithSpan};

pub type SurfPtr = Rc<WithSpan<Surface>>;

#[derive(Debug, Clone)]
pub enum Surface {
    Hole,
    Var(Name),
    Lambda(Name, Either<Icit, Name>, Option<SurfPtr>, SurfPtr),
    App(SurfPtr, SurfPtr, Either<Icit, Name>),
    AppPruning(SurfPtr, Pruning),
    Universe,
    Pi(Name, Icit, SurfPtr, SurfPtr),
    Let {
        name: Name,
        ty: SurfPtr,
        term: SurfPtr,
        body: SurfPtr,
    },
}
