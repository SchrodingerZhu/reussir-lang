#![allow(clippy::missing_safety_doc)]
#[no_mangle]
pub unsafe extern "C" fn __reussir_panic(ptr: *const u8, length: usize) -> ! {
    let message = std::slice::from_raw_parts(ptr, length);
    panic!("{}", std::str::from_utf8_unchecked(message))
}
