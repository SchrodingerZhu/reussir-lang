// RUN: %not %reussir-opt %s 2>&1 | %FileCheck %s
!test = !reussir.composite<{!reussir.composite<{f128, i32}>, i32}>
module @test {
    // CHECK: error: 'reussir.val2ref' op must return a reference whose pointee is of the same type of the input
    func.func @foo(%arg0 : !test) {
        %1 = reussir.val2ref %arg0 : !test -> !reussir.ref<f128, nonfreezing>
        return
    }
}
