// RUN: %reussir-opt %s | %FileCheck %s
!closure = !reussir.closure<(i32, i32) -> i32>
module @test {
    // CHECK: func.func @closure_test() -> !reussir.closure<(i32, i32) -> i32>
    func.func @closure_test() -> !closure {
        %1 = reussir.closure.new {
            ^bb(%arg0: i32, %arg1: i32):
                %2 = arith.addi %arg0, %arg1 : i32
                reussir.closure.yield %2 : i32
        } : !closure
        return %1 : !closure
    }
}
