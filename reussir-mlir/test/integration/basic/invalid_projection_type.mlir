// RUN: %not %reussir-opt %s 2>&1 | %FileCheck %s
!test = !reussir.composite<{!reussir.composite<{i32, i32, f128}>, i32}>
module @test {
    func.func @projection(%0: !reussir.rc<!test, nonatomic, nonfreezing>) {
        %1 = reussir.rc.borrow %0 : 
            !reussir.rc<!test, nonatomic, nonfreezing>
            -> !reussir.ref<!test, nonfreezing>
        // CHECK: error: 'reussir.proj' op expected to return a reference to '!reussir.composite<{i32, i32, f128}>', but found a reference to '!reussir.composite<{i32, i32}>'
        %2 = reussir.proj %1[0] : 
            !reussir.ref<!test, nonfreezing> -> !reussir.ref<!reussir.composite<{i32, i32}>, nonfreezing>
        return
    }
}
