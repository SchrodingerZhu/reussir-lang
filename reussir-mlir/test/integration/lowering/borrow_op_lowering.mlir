// RUN: %reussir-opt %s 2>&1 -convert-reussir-to-llvm | %FileCheck %s
!test = !reussir.composite<{!reussir.composite<{f128, i32}>, i32}>
module @test attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<f80, dense<128> : vector<2xi64>>, #dlti.dl_entry<i128, dense<128> : vector<2xi64>>, #dlti.dl_entry<i32, dense<32> : vector<2xi64>>, #dlti.dl_entry<f128, dense<128> : vector<2xi64>>, #dlti.dl_entry<f64, dense<64> : vector<2xi64>>, #dlti.dl_entry<f16, dense<16> : vector<2xi64>>, #dlti.dl_entry<i1, dense<8> : vector<2xi64>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi64>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi64>>, #dlti.dl_entry<i8, dense<8> : vector<2xi64>>, #dlti.dl_entry<i16, dense<16> : vector<2xi64>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi64>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi64>>, #dlti.dl_entry<i64, dense<64> : vector<2xi64>>, #dlti.dl_entry<"dlti.stack_alignment", 128 : i64>, #dlti.dl_entry<"dlti.endianness", "little">>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"} {
    func.func @borrow_nonfreezing(%0: !reussir.rc<!test, nonatomic, nonfreezing>) -> !reussir.ref<!test, nonfreezing> {
        // CHECK: %{{[0-9a-z]+}} = llvm.getelementptr %{{[0-9a-z]+}}[0, 2] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(i64, array<8 x i8>, struct<(struct<(f128, i32, array<12 x i8>)>, i32, array<12 x i8>)>)>
        %1 = reussir.rc.borrow %0 : 
            !reussir.rc<!test, nonatomic, nonfreezing>
            -> !reussir.ref<!test, nonfreezing>
        return %1 : !reussir.ref<!test, nonfreezing>
    }
    func.func @borrow_frozen(%0: !reussir.rc<!test, nonatomic, frozen>) -> !reussir.ref<!test, frozen> {
        // CHECK: %{{[0-9a-z]+}} = llvm.getelementptr %{{[0-9a-z]+}}[0, 4] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(i64, ptr, ptr, array<8 x i8>, struct<(struct<(f128, i32, array<12 x i8>)>, i32, array<12 x i8>)>)>
        %1 = reussir.rc.borrow %0 : 
            !reussir.rc<!test, nonatomic, frozen>
            -> !reussir.ref<!test, frozen>
        return %1 : !reussir.ref<!test, frozen>
    }
}
