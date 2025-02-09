// RUN: %not %reussir-opt %s 2>&1 | %FileCheck %s
module @test {
    // CHECK: error: size must be a multiple of alignment
    func.func @foo() {
        %1 = reussir.token.alloc : !reussir.token<size : 129, alignment : 16>
        reussir.token.free (%1 : !reussir.token<size: 129, alignment: 16>)
        return
    }
}
