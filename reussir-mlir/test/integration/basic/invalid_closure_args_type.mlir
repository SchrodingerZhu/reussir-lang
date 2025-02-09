// RUN: %not %reussir-opt %s 2>&1 | %FileCheck %s
!closure = !reussir.closure<(index) -> index>
module @test {
    func.func @closure_test() -> !closure {
        // CHECK: error: 'reussir.closure.new' op the types of arguments in the region must match the input types in the closure type
        %1 = reussir.closure.new {
            ^bb(%arg0: i32):
                %1 = arith.index_castui %arg0 : i32 to index
                reussir.closure.yield %1 : index
        } : !closure
        return %1 : !closure
    }
}
