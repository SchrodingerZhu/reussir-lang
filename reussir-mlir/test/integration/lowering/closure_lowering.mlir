// RUN: %reussir-opt %s -reussir-closure-outlining | %FileCheck %s --check-prefix=CHECK-OUTLINING
// RUN: %reussir-opt %s -reussir-closure-outlining -convert-reussir-to-llvm | %FileCheck %s --check-prefix=CHECK-LOWERING
!closure = !reussir.closure<(i32, i128) -> i128>
module @test attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<f80, dense<128> : vector<2xi64>>, #dlti.dl_entry<i128, dense<128> : vector<2xi64>>, #dlti.dl_entry<i32, dense<32> : vector<2xi64>>, #dlti.dl_entry<f128, dense<128> : vector<2xi64>>, #dlti.dl_entry<f64, dense<64> : vector<2xi64>>, #dlti.dl_entry<f16, dense<16> : vector<2xi64>>, #dlti.dl_entry<i1, dense<8> : vector<2xi64>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi64>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi64>>, #dlti.dl_entry<i8, dense<8> : vector<2xi64>>, #dlti.dl_entry<i16, dense<16> : vector<2xi64>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi64>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi64>>, #dlti.dl_entry<i64, dense<64> : vector<2xi64>>, #dlti.dl_entry<"dlti.stack_alignment", 128 : i64>, #dlti.dl_entry<"dlti.endianness", "little">>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"} 
{
    // CHECK-OUTLINING: func.func private @closure_test$$lambda0$$func(%arg0: !reussir.ref<!reussir.composite<{i32, i128}>, nonfreezing>) -> i128 {
    // CHECK-OUTLINING:   %[[REG0:[0-9a-z]+]] = reussir.proj %arg0[0] : <!reussir.composite<{i32, i128}>, nonfreezing> -> <i32, nonfreezing>
    // CHECK-OUTLINING:   %[[REG1:[0-9a-z]+]] = reussir.load %[[REG0]] : <i32, nonfreezing> -> i32
    // CHECK-OUTLINING:   %[[REG2:[0-9a-z]+]] = reussir.proj %arg0[1] : <!reussir.composite<{i32, i128}>, nonfreezing> -> <i128, nonfreezing>
    // CHECK-OUTLINING:   %[[REG3:[0-9a-z]+]] = reussir.load %[[REG2]] : <i128, nonfreezing> -> i128
    // CHECK-OUTLINING:   %[[REG4:[0-9a-z]+]] = arith.extui %[[REG1]] : i32 to i128
    // CHECK-OUTLINING:   %[[REG5:[0-9a-z]+]] = arith.addi %[[REG4]], %[[REG3]] : i128
    // CHECK-OUTLINING:   return %[[REG5]] : i128
    // CHECK-OUTLINING: }
    // TODO: check drop
    // CHECK-OUTLINING: reussir.closure.vtable @closure_test$$lambda0$$vtable{closure_type : !reussir.closure<(i32, i128) -> i128>, func : @closure_test$$lambda0$$func, clone : @closure_test$$lambda0$$clone, drop : @closure_test$$lambda0$$drop}
    // CHECK-OUTLINING: func.func @closure_test() -> !reussir.closure<(i32, i128) -> i128> {
    // CHECK-OUTLINING:   %[[REG0:[0-9a-z]+]] = reussir.token.alloc : <size : 32, alignment : 16>
    // CHECK-OUTLINING:   %[[REG1:[0-9a-z]+]] = reussir.closure.assemble vtable(@closure_test$$lambda0$$vtable) argpack(%[[REG0]] : !reussir.token<size : 32, alignment : 16>) : <(i32, i128) -> i128>
    // CHECK-OUTLINING:   return %[[REG1]] : !reussir.closure<(i32, i128) -> i128>
    // CHECK-OUTLINING: }


    // CHECK-LOWERING: llvm.func @closure_test$$lambda0$$func(%arg0: !llvm.ptr) -> i128 attributes {sym_visibility = "private"} {
    // CHECK-LOWERING:   %[[REG0:[0-9a-z]+]] = llvm.getelementptr %arg0[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(i32, array<12 x i8>, i128)>
    // CHECK-LOWERING:   %[[REG1:[0-9a-z]+]] = llvm.load %[[REG0]] {alignment = 4 : i64} : !llvm.ptr -> i32
    // CHECK-LOWERING:   %[[REG2:[0-9a-z]+]] = llvm.getelementptr %arg0[0, 2] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(i32, array<12 x i8>, i128)>
    // CHECK-LOWERING:   %[[REG3:[0-9a-z]+]] = llvm.load %[[REG2]] {alignment = 16 : i64} : !llvm.ptr -> i128
    // CHECK-LOWERING:   %[[REG4:[0-9a-z]+]] = arith.extui %[[REG1]] : i32 to i128
    // CHECK-LOWERING:   %[[REG5:[0-9a-z]+]] = arith.addi %[[REG4]], %[[REG3]] : i128
    // CHECK-LOWERING:   llvm.return %[[REG5]] : i128
    // CHECK-LOWERING: }
    // TODO: llvm.func @closure_test$$lambda0$$clone(!llvm.ptr, i64) -> !llvm.ptr attributes {sym_visibility = "private"}
    // TODO: llvm.func @closure_test$$lambda0$$drop(!llvm.ptr, i64) attributes {sym_visibility = "private"}
    // CHECK-LOWERING: llvm.mlir.global internal constant @closure_test$$lambda0$$vtable() {addr_space = 0 : i32, alignment = 8 : i64, dso_local} : !llvm.struct<(ptr, ptr, ptr)> {
    // CHECK-LOWERING:   %[[REG0:[0-9a-z]+]] = llvm.mlir.undef : !llvm.struct<(ptr, ptr, ptr)>
    // CHECK-LOWERING:   %[[REG1:[0-9a-z]+]] = llvm.mlir.addressof @closure_test$$lambda0$$func : !llvm.ptr
    // CHECK-LOWERING:   %[[REG2:[0-9a-z]+]] = llvm.insertvalue %[[REG1]], %[[REG0]][0] : !llvm.struct<(ptr, ptr, ptr)> 
    // CHECK-LOWERING:   %[[REG3:[0-9a-z]+]] = llvm.mlir.addressof @closure_test$$lambda0$$clone : !llvm.ptr
    // CHECK-LOWERING:   %[[REG4:[0-9a-z]+]] = llvm.insertvalue %[[REG3]], %[[REG2]][1] : !llvm.struct<(ptr, ptr, ptr)> 
    // CHECK-LOWERING:   %[[REG5:[0-9a-z]+]] = llvm.mlir.addressof @closure_test$$lambda0$$drop : !llvm.ptr
    // CHECK-LOWERING:   %[[REG6:[0-9a-z]+]] = llvm.insertvalue %[[REG5]], %[[REG4]][2] : !llvm.struct<(ptr, ptr, ptr)> 
    // CHECK-LOWERING:   llvm.return %[[REG6]] : !llvm.struct<(ptr, ptr, ptr)>
    // CHECK-LOWERING: }
    // CHECK-LOWERING: llvm.func @closure_test() -> !llvm.struct<(ptr, ptr, i64)> {
    // CHECK-LOWERING:   %[[REG0:[0-9a-z]+]] = llvm.mlir.constant(32 : i64) : i64
    // CHECK-LOWERING:   %[[REG1:[0-9a-z]+]] = llvm.mlir.constant(16 : i64) : i64
    // CHECK-LOWERING:   %[[REG2:[0-9a-z]+]] = llvm.call @__reussir_alloc(%[[REG0]], %[[REG1]]) : (i64, i64) -> !llvm.ptr
    // CHECK-LOWERING:   %[[REG3:[0-9a-z]+]] = llvm.mlir.undef : !llvm.struct<(ptr, ptr, i64)>
    // CHECK-LOWERING:   %[[REG4:[0-9a-z]+]] = llvm.mlir.constant(0 : i64) : i64
    // CHECK-LOWERING:   %[[REG5:[0-9a-z]+]] = llvm.mlir.addressof @closure_test$$lambda0$$vtable : !llvm.ptr
    // CHECK-LOWERING:   %[[REG6:[0-9a-z]+]] = llvm.insertvalue %[[REG5]], %[[REG3]][0] : !llvm.struct<(ptr, ptr, i64)> 
    // CHECK-LOWERING:   %[[REG7:[0-9a-z]+]] = llvm.insertvalue %[[REG2]], %[[REG6]][1] : !llvm.struct<(ptr, ptr, i64)> 
    // CHECK-LOWERING:   %[[REG8:[0-9a-z]+]] = llvm.insertvalue %[[REG4]], %[[REG7]][2] : !llvm.struct<(ptr, ptr, i64)> 
    // CHECK-LOWERING:   llvm.return %[[REG8]] : !llvm.struct<(ptr, ptr, i64)>
    // CHECK-LOWERING: }
    func.func @closure_test() -> !closure {
        %0 = reussir.closure.new {
            ^bb(%arg0: i32, %arg1: i128):
                %1 = arith.extui %arg0 : i32 to i128
                %2 = arith.addi %1, %arg1 : i128
                reussir.closure.yield %2 : i128
        } : !closure
        return %0 : !closure
    }
}
