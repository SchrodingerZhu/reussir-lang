// RUN: %not %reussir-opt %s 2>&1 | %FileCheck %s
module @test {
    // CHECK: error: 'reussir.rc.release' op cannot have any result when applied to a frozen RC pointer
    func.func @foo(%0 : !reussir.rc<i64, nonatomic, frozen>) {
        %1 = reussir.rc.release (%0 : !reussir.rc<i64, nonatomic, frozen>) : !reussir.nullable<!reussir.token<size: 8, alignment: 8>>
        return
    }
}
