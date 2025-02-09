#![allow(clippy::missing_safety_doc)]

use std::{ffi::c_void, ptr::NonNull};

use smallvec::SmallVec;

const STATUS_COUNTER_PADDING_BITS: usize = 2;
#[repr(C)]
pub struct RegionCtx {
    pub(crate) tail: *mut FreezableRcBoxHeader,
}

#[repr(C)]
#[derive(Clone, Copy)]
pub union FreezingStatus {
    ptr: *mut FreezableRcBoxHeader,
    scalar: usize,
}

#[derive(PartialEq, Eq)]
pub enum StatusKind {
    Unmarked,
    Representative,
    Rank,
    Rc,
    Disposing,
}

impl FreezingStatus {
    pub fn get_kind(self) -> StatusKind {
        unsafe {
            if self.scalar == 0 {
                return StatusKind::Unmarked;
            }
            if self.scalar as isize > 0 {
                return StatusKind::Representative;
            }
            match self.scalar & 0b11 {
                0 => StatusKind::Rank,
                1 => StatusKind::Rc,
                _ => StatusKind::Disposing,
            }
        }
    }
    pub fn get_value(self) -> usize {
        unsafe { (self.scalar & (isize::MAX as usize)) >> 2 }
    }
    pub unsafe fn get_pointer_unchecked(self) -> NonNull<FreezableRcBoxHeader> {
        unsafe { NonNull::new_unchecked(self.ptr) }
    }
    pub fn unmarked() -> Self {
        Self { scalar: 0 }
    }
    pub fn representative(ptr: *mut FreezableRcBoxHeader) -> Self {
        Self { ptr }
    }
    pub fn rank(value: usize) -> Self {
        let mask = isize::MIN as usize;
        Self {
            scalar: mask | (value << STATUS_COUNTER_PADDING_BITS),
        }
    }
    pub fn rc(value: usize) -> Self {
        let mask = isize::MIN as usize;
        Self {
            scalar: mask | (value << STATUS_COUNTER_PADDING_BITS) | 0b01,
        }
    }
    pub fn disposing() -> Self {
        let mask = isize::MIN as usize;
        Self {
            scalar: mask | 0b10,
        }
    }
}

type ActionFn = Option<unsafe extern "C" fn(*mut FreezableRcBoxHeader, *mut c_void)>;
type ScannerFn = Option<unsafe extern "C" fn(*mut c_void, ActionFn, *mut c_void)>;
type DropFn = Option<unsafe extern "C" fn(*mut c_void)>;

#[repr(C)]
pub struct FreezableVTable {
    drop: DropFn,
    scanner: ScannerFn,
    size: usize,
    alignment: usize,
    data_offset: usize,
}

#[repr(C)]
pub struct FreezableRcBoxHeader {
    status: FreezingStatus,
    next: *mut Self,
    vtable: NonNull<FreezableVTable>,
}

unsafe fn increase_refcnt(mut object: NonNull<FreezableRcBoxHeader>) {
    object.as_mut().status.scalar += 1 << STATUS_COUNTER_PADDING_BITS;
}

unsafe fn decrease_refcnt(mut object: NonNull<FreezableRcBoxHeader>) -> bool {
    object.as_mut().status.scalar -= 1 << STATUS_COUNTER_PADDING_BITS;
    object.as_mut().status.scalar <= (isize::MIN as usize | 0b11)
}

unsafe fn find_representative(
    mut object: NonNull<FreezableRcBoxHeader>,
) -> NonNull<FreezableRcBoxHeader> {
    let mut root = object;
    while root.as_ref().status.get_kind() == StatusKind::Representative {
        root = root.as_ref().status.get_pointer_unchecked();
    }
    while object.as_ref().status.get_kind() == StatusKind::Representative {
        let next = object.as_ref().status.get_pointer_unchecked();
        object.as_mut().status = FreezingStatus::representative(root.as_ptr());
        object = next;
    }
    root
}

unsafe fn union(x: NonNull<FreezableRcBoxHeader>, y: NonNull<FreezableRcBoxHeader>) -> bool {
    let mut rep_x = find_representative(x);
    let mut rep_y = find_representative(y);
    if rep_x == rep_y {
        return false;
    }
    if rep_x.as_ref().status.get_value() < rep_y.as_ref().status.get_value() {
        std::mem::swap(&mut rep_x, &mut rep_y);
    }
    if rep_x.as_ref().status.get_value() == rep_y.as_ref().status.get_value() {
        rep_x.as_mut().status = FreezingStatus::rank(rep_x.as_ref().status.get_value() + 1);
    }
    rep_y.as_mut().status = FreezingStatus::representative(rep_x.as_ptr());
    true
}

#[cold]
unsafe fn dispose(object: NonNull<FreezableRcBoxHeader>) {
    type Stack = SmallVec<[NonNull<FreezableRcBoxHeader>; 16]>;
    unsafe fn add_stack(stack: &mut Stack, mut object: NonNull<FreezableRcBoxHeader>) {
        stack.push(object);
        object.as_mut().status = FreezingStatus::disposing();
    }
    struct DisposeContext {
        dfs: Stack,
        scc: Stack,
        recycle: Stack,
    }
    let mut ctx = DisposeContext {
        dfs: Stack::new(),
        scc: Stack::new(),
        recycle: Stack::new(),
    };
    add_stack(&mut ctx.dfs, find_representative(object));
    while let Some(obj) = ctx.dfs.pop() {
        ctx.scc.push(obj);
        while let Some(obj) = ctx.scc.pop() {
            ctx.recycle.push(obj);
            unsafe extern "C" fn dispose_action(
                field: *mut FreezableRcBoxHeader,
                ctx: *mut c_void,
            ) {
                let ctx = &mut *(ctx as *mut DisposeContext);
                let field = NonNull::new_unchecked(field);
                let next = find_representative(field);
                match next.as_ref().status.get_kind() {
                    StatusKind::Disposing if field != next => add_stack(&mut ctx.scc, field),
                    StatusKind::Rc if decrease_refcnt(next) => add_stack(&mut ctx.dfs, next),
                    _ => (),
                }
            }
            let vtable = obj.as_ref().vtable;
            if let Some(scanner) = vtable.as_ref().scanner {
                scanner(
                    obj.as_ptr()
                        .cast::<c_void>()
                        .byte_add(vtable.as_ref().data_offset),
                    Some(dispose_action),
                    &mut ctx as *mut _ as *mut c_void,
                );
            }
        }
    }
    stacker::maybe_grow(16 * 1024, 1024 * 1024, || {
        while let Some(obj) = ctx.recycle.pop() {
            let vtable = obj.as_ref().vtable;
            if let Some(dtor) = vtable.as_ref().drop {
                let ptr = obj
                    .as_ptr()
                    .cast::<c_void>()
                    .byte_add(vtable.as_ref().data_offset);
                dtor(ptr);
            }
            crate::allocator::__reussir_dealloc(
                obj.cast().as_ptr(),
                vtable.as_ref().size,
                vtable.as_ref().alignment,
            );
        }
    });
}

#[no_mangle]
pub unsafe extern "C" fn __reussir_freeze(object: *mut FreezableRcBoxHeader) {
    let Some(object) = NonNull::new(object) else {
        panic!("attempt to freeze null object");
    };
    type PendingList = SmallVec<[NonNull<FreezableRcBoxHeader>; 32]>;
    let mut pending = PendingList::new();
    unsafe extern "C" fn freeze(object: *mut FreezableRcBoxHeader, pending: *mut c_void) {
        let pending = &mut *(pending as *mut PendingList);
        let mut object = NonNull::new_unchecked(object);
        match object.as_ref().status.get_kind() {
            StatusKind::Rc => {
                let root = find_representative(object);
                increase_refcnt(root);
            }
            StatusKind::Rank => loop {
                let Some(next) = pending.pop() else {
                    break;
                };
                if !union(object, next) {
                    break;
                }
            },
            StatusKind::Unmarked => {
                object.as_mut().status = FreezingStatus::rc(1);
                pending.push(object);
                stacker::maybe_grow(16 * 1024, 1024 * 1024, || {
                    let vtable = object.as_ref().vtable;
                    if let Some(scanner) = vtable.as_ref().scanner {
                        scanner(
                            object
                                .as_ptr()
                                .cast::<c_void>()
                                .byte_add(vtable.as_ref().data_offset),
                            Some(freeze),
                            pending as *mut _ as *mut c_void,
                        );
                    }
                });
            }
            _ => std::hint::unreachable_unchecked(),
        }
    }
    freeze(object.as_ptr(), (&mut pending) as *mut _ as *mut c_void);
}

#[no_mangle]
pub unsafe extern "C" fn __reussir_freeze_atomic(object: *mut FreezableRcBoxHeader) {
    unimplemented!("cannot freeze {object:?}")
}

#[no_mangle]
pub unsafe extern "C" fn __reussir_acquire_freezable(object: *mut FreezableRcBoxHeader) {
    let Some(object) = NonNull::new(object) else {
        panic!("attempt to acquire null object");
    };
    let root = find_representative(object);
    increase_refcnt(root);
}

#[no_mangle]
pub unsafe extern "C" fn __reussir_release_freezable(object: *mut FreezableRcBoxHeader) {
    if let Some(object) = NonNull::new(object) {
        if decrease_refcnt(object) {
            dispose(object);
        }
    } else {
        panic!("attempt to release null object");
    }
}

#[no_mangle]
pub unsafe extern "C" fn __reussir_acquire_atomic_freezable(object: *mut FreezableRcBoxHeader) {
    unimplemented!("cannot acquire {object:?}")
}

#[no_mangle]
pub unsafe extern "C" fn __reussir_release_atomic_freezable(object: *mut FreezableRcBoxHeader) {
    unimplemented!("cannot release {object:?}")
}

#[no_mangle]
pub unsafe extern "C" fn __reussir_clean_up_region(region: *mut RegionCtx) {
    let mut tail = NonNull::new((*region).tail);
    while let Some(object) = tail {
        tail = NonNull::new(object.as_ref().next);
        if matches!(object.as_ref().status.get_kind(), StatusKind::Unmarked) {
            let vtable = object.as_ref().vtable;
            if let Some(dtor) = vtable.as_ref().drop {
                let ptr = object
                    .as_ptr()
                    .cast::<c_void>()
                    .byte_add(vtable.as_ref().data_offset);
                dtor(ptr);
            }
            crate::allocator::__reussir_dealloc(
                object.cast().as_ptr(),
                vtable.as_ref().size,
                vtable.as_ref().alignment,
            );
        }
    }
}

#[no_mangle]
pub unsafe extern "C" fn __reussir_clean_up_region_atomic(region: *mut RegionCtx) {
    unimplemented!("cannot clean up region: {region:?}")
}
