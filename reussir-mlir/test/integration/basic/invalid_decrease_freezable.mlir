// RUN: %not %reussir-opt %s 2>&1 | %FileCheck %s
module @test {
    // CHECK: error: 'reussir.rc.decrease' op can only be applied to a nonfreezing RC pointer
    func.func @foo(%0 : !reussir.rc<i64, nonatomic, frozen>) {
        %1 = reussir.rc.decrease (%0 : !reussir.rc<i64, nonatomic, frozen>) : i1
        return
    }
}
