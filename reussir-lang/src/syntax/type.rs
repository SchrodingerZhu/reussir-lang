#![allow(unused)]
pub enum Type<'ctx> {
    Int(Int),
    Float(Float),
    Str,
    Composite(&'ctx [Self]),
}

pub struct Int {
    width_ilog: u32,
    signed: bool,
}

impl Int {
    pub fn new(width: usize, signed: bool) -> Self {
        assert!(
            width.is_power_of_two(),
            "width must be a power of 2, found {}.",
            width
        );
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

pub enum Float {
    F16,
    BF16,
    F32,
    F64,
    F128,
}

pub enum FieldName<'ctx> {
    Idx(usize),
    Name(&'ctx str),
}

pub struct Composite<'ctx> {
    name: &'ctx str,
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    #[should_panic]
    fn int_rejects_invalid_width() {
        _ = Int::new(13, true);
    }

    #[test]
    fn int_correctly_records_width() {
        assert_eq!(Int::new(16, true).width(), 16);
    }
}
