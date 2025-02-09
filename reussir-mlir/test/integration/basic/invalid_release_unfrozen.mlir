// RUN: %not %reussir-opt %s 2>&1 | %FileCheck %s
module @test {
    // CHECK: error: 'reussir.rc.release' op cannot be applied to an unfrozen RC pointer
    func.func @foo(%0 : !reussir.rc<i64, nonatomic, unfrozen>) {
        reussir.rc.release (%0 : !reussir.rc<i64, nonatomic, unfrozen>)
        return
    }
}
