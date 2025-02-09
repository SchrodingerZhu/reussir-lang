// RUN: %reussir-opt %s 2>&1
!incomplete = !reussir.composite<"test" incomplete>
!test = !reussir.composite<"test" {i32, !reussir.mref<!incomplete, nonatomic>}>

module @test {
    func.func @foo(%0: !reussir.rc<!test, nonatomic, frozen>) {
        %1 = reussir.rc.borrow %0 : 
            !reussir.rc<!test, nonatomic, frozen>
            -> !reussir.ref<!test, frozen>
        %2 = reussir.proj %1[1] : 
            !reussir.ref<!test, frozen> -> !reussir.ref<!reussir.mref<!test, nonatomic>, frozen>
        %3 = reussir.load %2 : !reussir.ref<!reussir.mref<!test, nonatomic>, frozen> -> !reussir.nullable<!reussir.rc<!test, nonatomic, frozen>>
        return
    }
}
