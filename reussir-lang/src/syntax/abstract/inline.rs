use std::collections::HashMap;

use crate::syntax::ID;
use crate::syntax::r#abstract::Term;

#[allow(dead_code)]
#[derive(Default)]
pub struct Inliner<'src> {
    env: HashMap<ID, Term<'src>>,
}

#[allow(dead_code)]
impl<'src> Inliner<'src> {
    pub fn with(mut self, id: ID, typ: Term<'src>) -> Self {
        self.env.insert(id, typ);
        self
    }

    pub fn term(&mut self, tm: &mut Term<'src>) {
        use Term::*;
        match tm {
            Ident(i) => {
                if let Some(v) = self.env.get(&i.id) {
                    *tm = v.clone();
                }
            }
            FnType {
                param_types,
                eff,
                ret,
            } => {
                param_types.iter_mut().for_each(|t| self.term(t));
                self.term(eff);
                self.term(ret);
            }
            Fn { body, .. } => self.term(body),
            Call { f, args } => {
                self.term(f);
                args.iter_mut().for_each(|a| self.term(a));
            }
            GenericFnType { param, body } | GenericFn { param, body } => {
                self.term(&mut param.typ);
                self.term(body);
            }
            _ => {}
        }
    }
}
