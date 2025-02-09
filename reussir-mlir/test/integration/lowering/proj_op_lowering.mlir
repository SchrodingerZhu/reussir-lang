// RUN: %reussir-opt %s -convert-reussir-to-llvm | %FileCheck %s
!test = !reussir.composite<{!reussir.composite<{i32, i32, f128}>, i32}>
!array = !reussir.array<!test, 16, 16>
module @test attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<f80, dense<128> : vector<2xi64>>, #dlti.dl_entry<i128, dense<128> : vector<2xi64>>, #dlti.dl_entry<i32, dense<32> : vector<2xi64>>, #dlti.dl_entry<f128, dense<128> : vector<2xi64>>, #dlti.dl_entry<f64, dense<64> : vector<2xi64>>, #dlti.dl_entry<f16, dense<16> : vector<2xi64>>, #dlti.dl_entry<i1, dense<8> : vector<2xi64>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi64>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi64>>, #dlti.dl_entry<i8, dense<8> : vector<2xi64>>, #dlti.dl_entry<i16, dense<16> : vector<2xi64>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi64>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi64>>, #dlti.dl_entry<i64, dense<64> : vector<2xi64>>, #dlti.dl_entry<"dlti.stack_alignment", 128 : i64>, #dlti.dl_entry<"dlti.endianness", "little">>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"} 
{
    func.func @composite_projection(%0: !reussir.rc<!test, nonatomic, nonfreezing>) {
        %1 = reussir.rc.borrow %0 : 
            !reussir.rc<!test, nonatomic, nonfreezing>
            -> !reussir.ref<!test, nonfreezing>
        // CHECK: %[[REG0:[0-9a-z]+]] = llvm.getelementptr %{{[0-9a-z]+}}[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(struct<(i32, i32, array<8 x i8>, f128)>, i32, array<12 x i8>)>
        %2 = reussir.proj %1[0] : 
            !reussir.ref<!test, nonfreezing> -> !reussir.ref<!reussir.composite<{i32, i32, f128}>, nonfreezing>
        // CHECK: %{{[0-9a-z]+}} = llvm.getelementptr %[[REG0]][0, 3] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(i32, i32, array<8 x i8>, f128)>
        %3 = reussir.proj %2[2] : 
            !reussir.ref<!reussir.composite<{i32, i32, f128}>, nonfreezing> -> !reussir.ref<f128, nonfreezing>
        return
    }
    func.func @array_projection(%0: !reussir.rc<!array, nonatomic, nonfreezing>) {
        %1 = reussir.rc.borrow %0 : 
            !reussir.rc<!array, nonatomic, nonfreezing>
            -> !reussir.ref<!array, nonfreezing>
        // CHECK: %[[REG0:[0-9a-z]+]] = llvm.getelementptr %{{[0-9a-z]+}}[13] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<16 x struct<(struct<(i32, i32, array<8 x i8>, f128)>, i32, array<12 x i8>)>>
        %2 = reussir.proj %1[13] : 
            !reussir.ref<!array, nonfreezing> -> !reussir.ref<!reussir.array<!test, 16>, nonfreezing>
        // CHECK: %[[REG1:[0-9a-z]+]] = llvm.getelementptr %[[REG0]][7] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(struct<(i32, i32, array<8 x i8>, f128)>, i32, array<12 x i8>)>
        %3 = reussir.proj %2[7] : 
            !reussir.ref<!reussir.array<!test, 16>, nonfreezing> -> !reussir.ref<!test, nonfreezing>
        // CHECK: %{{[0-9a-z]+}} = llvm.getelementptr %[[REG1]][0, 1] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(struct<(i32, i32, array<8 x i8>, f128)>, i32, array<12 x i8>)>
        %4 = reussir.proj %3[1] : 
            !reussir.ref<!test, nonfreezing>-> !reussir.ref<i32, nonfreezing>
        return
    }
}
