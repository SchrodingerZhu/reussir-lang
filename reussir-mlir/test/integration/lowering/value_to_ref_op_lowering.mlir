// RUN: %reussir-opt %s 2>&1 -convert-reussir-to-llvm | %FileCheck %s --check-prefix='CHECK-ORIGIN'
!test = !reussir.composite<{!reussir.composite<{i128, i32}>, i32}>
module @test attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<f80, dense<128> : vector<2xi64>>, #dlti.dl_entry<i128, dense<128> : vector<2xi64>>, #dlti.dl_entry<i32, dense<32> : vector<2xi64>>, #dlti.dl_entry<f128, dense<128> : vector<2xi64>>, #dlti.dl_entry<f64, dense<64> : vector<2xi64>>, #dlti.dl_entry<f16, dense<16> : vector<2xi64>>, #dlti.dl_entry<i1, dense<8> : vector<2xi64>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi64>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi64>>, #dlti.dl_entry<i8, dense<8> : vector<2xi64>>, #dlti.dl_entry<i16, dense<16> : vector<2xi64>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi64>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi64>>, #dlti.dl_entry<i64, dense<64> : vector<2xi64>>, #dlti.dl_entry<"dlti.stack_alignment", 128 : i64>, #dlti.dl_entry<"dlti.endianness", "little">>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"} {
    func.func @borrow_nonfreezing(%0: !test) {
        // CHECK-ORIGIN: %[[REG0:[a-z0-9]+]] = llvm.mlir.constant(1 : i64) : i64
        // CHECK-ORIGIN: %[[REG1:[a-z0-9]+]] = llvm.alloca %[[REG0]] x !llvm.struct<(struct<(i128, i32, array<12 x i8>)>, i32, array<12 x i8>)> {alignment = 16 : i64} : (i64) -> !llvm.ptr
        // CHECK-ORIGIN: llvm.store %{{[a-z0-9]+}}, %[[REG1]] {alignment = 16 : i64} : !llvm.struct<(struct<(i128, i32, array<12 x i8>)>, i32, array<12 x i8>)>, !llvm.ptr
        %1 = reussir.val2ref %0 : !test -> !reussir.ref<!test, nonfreezing>
        func.call @opaque (%1) : (!reussir.ref<!test, nonfreezing>) -> ()
        return
    }
    func.func private @opaque(%0: !reussir.ref<!test, nonfreezing>)
}
