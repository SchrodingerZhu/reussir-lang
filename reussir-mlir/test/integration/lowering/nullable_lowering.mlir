// RUN: %reussir-opt %s -convert-reussir-to-llvm | %FileCheck %s
!rc = !reussir.rc<i64, nonatomic, frozen>
module @test {
    //CHECK: llvm.func @foo(%[[arg0:[0-9a-z]+]]: !llvm.ptr) -> !llvm.ptr {
    //CHECK:     llvm.return %[[arg0]] : !llvm.ptr
    //CHECK: }
    func.func @foo(%arg0: !rc) -> !reussir.nullable<!rc> {
        %0 = reussir.nullable.nonnull(%arg0 : !rc) : !reussir.nullable<!rc>
        return %0 : !reussir.nullable<!rc>
    }
    //CHECK: llvm.func @bar() -> !llvm.ptr {
    //CHECK:     %[[val:[0-9a-z]+]] = llvm.mlir.zero : !llvm.ptr
    //CHECK:     llvm.return %[[val]] : !llvm.ptr
    //CHECK: }
    func.func @bar() -> !reussir.nullable<!rc> {
        %0 = reussir.nullable.null : !reussir.nullable<!rc>
        return %0 : !reussir.nullable<!rc>
    }
}
