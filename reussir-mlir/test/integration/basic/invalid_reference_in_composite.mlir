// RUN: %not %reussir-opt %s 2>&1 | %FileCheck %s
// CHECK: error: cannot have a reference type in a composite type
!test = !reussir.composite<{!reussir.ref<i32, nonfreezing>, i32}>
