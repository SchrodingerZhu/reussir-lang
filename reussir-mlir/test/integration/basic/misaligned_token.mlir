// RUN: %not %reussir-opt %s 2>&1 | %FileCheck %s
module @test {
    // CHECK: error: alignment must be a power of 2
    func.func @foo(%0 : !reussir.rc<i64, nonatomic, nonfreezing>, %test: !llvm.struct<()>) {
        reussir.rc.acquire (%0 : !reussir.rc<i64, nonatomic, nonfreezing>)
        %1 = reussir.token.alloc : !reussir.token<size : 128, alignment : 13>
        reussir.token.free (%1 : !reussir.token<size: 128, alignment: 13>)
        return
    }
}
