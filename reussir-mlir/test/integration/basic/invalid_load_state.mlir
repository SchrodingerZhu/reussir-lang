// RUN: %not %reussir-opt %s 2>&1 | %FileCheck %s
module @test {
    func.func @load(%0: !reussir.ref<!reussir.mref<f128, nonatomic>, nonfreezing>) -> !reussir.rc<f128, nonatomic, nonfreezing> {
        // CHECK: error: 'reussir.load' op cannot load a mutable RC pointer through a nonfreezing reference
        %1 = reussir.load %0 : !reussir.ref<!reussir.mref<f128, nonatomic>, nonfreezing> -> !reussir.rc<f128, nonatomic, nonfreezing>
        return %1 : !reussir.rc<f128, nonatomic, nonfreezing>
    }
}
