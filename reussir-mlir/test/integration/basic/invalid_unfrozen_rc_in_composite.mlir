// RUN: %not %reussir-opt %s 2>&1 | %FileCheck %s
// CHECK: error: cannot have a non-frozen but freezable RC type in a composite type, use mref instead
!test = !reussir.composite<{!reussir.rc<i32, nonatomic, unfrozen>, i32}>
