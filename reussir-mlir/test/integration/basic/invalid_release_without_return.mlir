// RUN: %not %reussir-opt %s 2>&1 | %FileCheck %s
module @test {
    // CHECK: error: 'reussir.rc.release' op must have a result when applied to a nonfreezing RC pointer
    func.func @foo(%0 : !reussir.rc<i64, nonatomic, nonfreezing>) {
        reussir.rc.release (%0 : !reussir.rc<i64, nonatomic, nonfreezing>)
        return
    }
}
