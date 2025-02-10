#![allow(unused)]
use bumpalo::Bump;

mod expr;
mod func;
mod r#type;

struct Context {
    arena: Bump,
    input: std::path::PathBuf,
    src: String,
}
