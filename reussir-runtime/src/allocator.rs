#![allow(clippy::missing_safety_doc)]
use std::{
    alloc::{GlobalAlloc, Layout},
    ffi::c_void,
};

#[cfg(feature = "snmalloc")]
extern "C" {
    pub fn __reussir_alloc_impl(size: usize, alignment: usize) -> *mut c_void;
    pub fn __reussir_dealloc_impl(ptr: *mut c_void, size: usize, alignment: usize);
    pub fn __reussir_realloc_impl(
        ptr: *mut c_void,
        old_size: usize,
        alignment: usize,
        size: usize,
    ) -> *mut c_void;
    pub fn __reussir_realloc_nocopy_impl(
        ptr: *mut c_void,
        old_size: usize,
        alignment: usize,
        size: usize,
    ) -> *mut c_void;
}

#[no_mangle]
pub unsafe extern "C" fn __reussir_alloc(size: usize, alignment: usize) -> *mut c_void {
    let res = if cfg!(feature = "snmalloc") {
        __reussir_alloc_impl(size, alignment)
    } else {
        std::alloc::alloc(Layout::from_size_align_unchecked(size, alignment)) as _
    };
    if res.is_null() {
        std::alloc::handle_alloc_error(Layout::from_size_align_unchecked(size, alignment));
    }
    res
}

#[no_mangle]
pub unsafe extern "C" fn __reussir_dealloc(ptr: *mut c_void, size: usize, alignment: usize) {
    if cfg!(feature = "snmalloc") {
        __reussir_dealloc_impl(ptr, size, alignment);
    } else {
        std::alloc::dealloc(ptr as _, Layout::from_size_align_unchecked(size, alignment));
    }
}

#[no_mangle]
pub unsafe extern "C" fn __reussir_realloc(
    ptr: *mut c_void,
    old_size: usize,
    alignment: usize,
    size: usize,
) -> *mut c_void {
    let res = if cfg!(feature = "snmalloc") {
        __reussir_realloc_impl(ptr, old_size, alignment, size)
    } else {
        std::alloc::realloc(
            ptr as _,
            Layout::from_size_align_unchecked(old_size, alignment),
            size,
        ) as _
    };
    if res.is_null() {
        std::alloc::handle_alloc_error(Layout::from_size_align_unchecked(size, alignment));
    }
    res
}

#[no_mangle]
pub unsafe extern "C" fn __reussir_realloc_nocopy(
    ptr: *mut c_void,
    old_size: usize,
    alignment: usize,
    size: usize,
) -> *mut c_void {
    let res = if cfg!(feature = "snmalloc") {
        __reussir_realloc_nocopy_impl(ptr, old_size, alignment, size)
    } else {
        std::alloc::realloc(
            ptr as _,
            Layout::from_size_align_unchecked(old_size, alignment),
            size,
        ) as _
    };
    if res.is_null() {
        std::alloc::handle_alloc_error(Layout::from_size_align_unchecked(size, alignment));
    }
    res as _
}

#[cfg(feature = "snmalloc")]
mod global {
    use super::*;
    struct SnMalloc;

    unsafe impl GlobalAlloc for SnMalloc {
        unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
            __reussir_alloc_impl(layout.size(), layout.align()) as _
        }

        unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
            __reussir_dealloc_impl(ptr as _, layout.size(), layout.align());
        }

        unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
            __reussir_realloc_impl(ptr as _, layout.size(), layout.align(), new_size) as _
        }
    }

    #[global_allocator]
    static A: SnMalloc = SnMalloc;
}
