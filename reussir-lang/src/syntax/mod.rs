use bumpalo::Bump;

mod expr;
mod func;
mod r#type;

struct Context {
    allocator: Bump,
    input: std::path::PathBuf,
}
