// RUN: %reussir-opt %s | %FileCheck %s
!test = !reussir.composite<{!reussir.composite<{i32, i32}>, i32}>
module @test {
    
    // CHECK: func.func @foo(%{{[0-9a-z]+}}: !reussir.rc<i64, atomic, nonfreezing>, %{{[0-9a-z]+}}: !llvm.struct<()>)
    func.func @foo(%0 : !reussir.rc<i64, atomic, nonfreezing>, %test: !llvm.struct<()>) {
        reussir.rc.acquire (%0 : !reussir.rc<i64, atomic, nonfreezing>)
        %1 = reussir.token.alloc : !reussir.token<size : 128, alignment : 16>
        reussir.token.free (%1 : !reussir.token<size: 128, alignment: 16>)
        return
    }
    // CHECK: func.func @bar(%{{[0-9a-z]+}}: !reussir.rc<i64, atomic, nonfreezing>)
    func.func @bar(%0 : !reussir.rc<i64, atomic, nonfreezing>) {
        return
    }
    // CHECK: func.func @baz(%{{[0-9a-z]+}}: !reussir.rc<i64, nonatomic, frozen>)
    func.func @baz(%0 : !reussir.rc<i64, nonatomic, frozen>) {
        return
    }
    // CHECK: func.func @qux(%{{[0-9a-z]+}}: !reussir.rc<i64, atomic, unfrozen>)
    func.func @qux(%0 : !reussir.rc<i64, atomic, unfrozen>) {
        return
    }
    func.func @projection(%0: !reussir.rc<!test, nonatomic, nonfreezing>) {
        %1 = reussir.rc.borrow %0 : 
            !reussir.rc<!test, nonatomic, nonfreezing>
            -> !reussir.ref<!test, nonfreezing>
        %2 = reussir.proj %1[0] : 
            !reussir.ref<!test, nonfreezing> -> !reussir.ref<!reussir.composite<{i32, i32}>, nonfreezing>
        %3 = reussir.proj %2[1] : 
            !reussir.ref<!reussir.composite<{i32, i32}>, nonfreezing> -> !reussir.ref<i32, nonfreezing> 
        return
    }
}
