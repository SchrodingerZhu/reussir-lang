// RUN: %not %reussir-opt %s 2>&1 | %FileCheck %s
module @test {
    // CHECK: error: the borrowed reference must have the consistent pointee type with the RC pointer
    func.func @borrow_state_invalid(%0: !reussir.rc<i32, nonatomic, frozen>) {
        %1 = reussir.rc.borrow %0 : 
            !reussir.rc<i32, nonatomic, frozen>
            -> !reussir.ref<i64, frozen>
        return
    }
}
