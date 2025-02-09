// RUN: %not %reussir-opt %s 2>&1 | %FileCheck %s
!closure = !reussir.closure<(i32, i32) -> i32>
module @test {
    func.func @closure_test() -> !closure {
        // CHECK: error: 'reussir.closure.yield' op expected to yield a value of 'i32', but 'index' is found instead
        %1 = reussir.closure.new {
            ^bb(%arg0: i32, %arg1: i32):
                %2 = arith.addi %arg0, %arg1 : i32
                %3 = arith.index_castui %2 : i32 to index
                reussir.closure.yield %3 : index
        } : !closure
        return %1 : !closure
    }
}
