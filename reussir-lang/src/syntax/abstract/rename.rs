use std::collections::HashMap;

use crate::syntax::r#abstract::Term;
use crate::syntax::{fresh, Ident, Param, ID};

pub fn rename(term: &mut Term) {
    Renamer::default().run(term)
}

#[derive(Default)]
struct Renamer(HashMap<ID, ID>);

impl Renamer {
    fn run(&mut self, term: &mut Term) {
        use Term::*;
        match term {
            Ident(i) => self.try_rename_ident(i),
            FnType {
                param_types,
                eff,
                ret,
            } => {
                param_types.iter_mut().for_each(|t| self.run(t));
                self.run(eff);
                self.run(ret);
            }
            Fn { params, body } => {
                params.iter_mut().for_each(|i| self.param_ident(i));
                self.run(body);
            }
            Call { f, args } => {
                self.run(f);
                args.iter_mut().for_each(|a| self.run(a));
            }
            GenericFnType { param, body } | GenericFn { param, body } => {
                self.param(param);
                self.run(body);
            }
            _ => {}
        }
    }

    fn param<'src>(&mut self, p: &mut Param<'src, Term<'src>>) {
        self.param_ident(&mut p.name);
        self.run(&mut p.typ);
    }

    fn param_ident(&mut self, ident: &mut Ident) {
        let old = ident.id;
        ident.id = fresh();
        self.0.insert(old, ident.id);
    }

    fn try_rename_ident(&mut self, ident: &mut Ident) {
        if let Some(id) = self.0.get(&ident.id) {
            ident.id = *id;
        }
    }
}
