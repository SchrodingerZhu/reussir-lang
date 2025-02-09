// RUN: %not %reussir-opt %s 2>&1 | %FileCheck %s
module @test {
    func.func @load(%0: !reussir.ref<f128, nonfreezing>) -> f32 {
        // CHECK: error: 'reussir.load' op expected to return a value of 'f128', but 'f32' is found instead
        %1 = reussir.load %0 : !reussir.ref<f128, nonfreezing> -> f32
        return %1 : f32
    }
}
