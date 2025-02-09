// RUN: %not %reussir-opt %s 2>&1 | %FileCheck %s
module @test {
    // CHECK: error: the borrowed reference must have the consistent freezing state with the RC pointer
    func.func @borrow_state_invalid(%0: !reussir.rc<i32, nonatomic, nonfreezing>) {
        %1 = reussir.rc.borrow %0 : 
            !reussir.rc<i32, nonatomic, nonfreezing>
            -> !reussir.ref<i32, frozen>
        return
    }
}
