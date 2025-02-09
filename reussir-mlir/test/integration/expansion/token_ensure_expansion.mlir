// RUN: %reussir-opt %s 2>&1 -reussir-expand-control-flow | %FileCheck %s
// RUN: %reussir-opt %s 2>&1 \
// RUN: -reussir-expand-control-flow \
// RUN: -convert-scf-to-cf \
// RUN: -convert-reussir-to-llvm |\
// RUN: %FileCheck %s --check-prefix=CHECK-LOWERING
!tk = !reussir.token<size: 64, alignment: 16>
module @test attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<f80, dense<128> : vector<2xi64>>, #dlti.dl_entry<i128, dense<128> : vector<2xi64>>, #dlti.dl_entry<i32, dense<32> : vector<2xi64>>, #dlti.dl_entry<f128, dense<128> : vector<2xi64>>, #dlti.dl_entry<f64, dense<64> : vector<2xi64>>, #dlti.dl_entry<f16, dense<16> : vector<2xi64>>, #dlti.dl_entry<i1, dense<8> : vector<2xi64>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi64>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi64>>, #dlti.dl_entry<i8, dense<8> : vector<2xi64>>, #dlti.dl_entry<i16, dense<16> : vector<2xi64>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi64>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi64>>, #dlti.dl_entry<i64, dense<64> : vector<2xi64>>, #dlti.dl_entry<"dlti.stack_alignment", 128 : i64>, #dlti.dl_entry<"dlti.endianness", "little">>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"} {
    // CHECK: func.func @token_ensure_expansion(%[[regarg0:[a-z0-9]+]]: !reussir.nullable<!reussir.token<size : 64, alignment : 16>>) -> !reussir.token<size : 64, alignment : 16> {
    // CHECK:     %[[reg0:[a-z0-9]+]] = reussir.nullable.check(%[[regarg0]] : <!reussir.token<size : 64, alignment : 16>>) -> i1
    // CHECK:     %[[reg1:[a-z0-9]+]] = scf.if %[[reg0]] -> (!reussir.token<size : 64, alignment : 16>) {
    // CHECK:     %[[reg2:[a-z0-9]+]] = reussir.nullable.coerce(%[[regarg0]] : <!reussir.token<size : 64, alignment : 16>>) : !reussir.token<size : 64, alignment : 16>
    // CHECK:     scf.yield %[[reg2]] : !reussir.token<size : 64, alignment : 16>
    // CHECK:     } else {
    // CHECK:     %[[reg2:[a-z0-9]+]] = reussir.token.alloc : <size : 64, alignment : 16>
    // CHECK:     scf.yield %[[reg2]] : !reussir.token<size : 64, alignment : 16>
    // CHECK:     }
    // CHECK:     return %[[reg1]] : !reussir.token<size : 64, alignment : 16>
    // CHECK: }
    
    //  CHECK-LOWERING: llvm.func @token_ensure_expansion(%[[regarg0:[a-z0-9]+]]: !llvm.ptr) -> !llvm.ptr {
    //  CHECK-LOWERING:   %[[reg0:[a-z0-9]+]] = llvm.mlir.zero : !llvm.ptr
    //  CHECK-LOWERING:   %[[reg1:[a-z0-9]+]] = llvm.icmp "ne" %[[regarg0]], %[[reg0]] : !llvm.ptr
    //  CHECK-LOWERING:   llvm.cond_br %[[reg1]], ^bb1, ^bb2
    //  CHECK-LOWERING: ^bb1:  // pred: ^bb0
    //  CHECK-LOWERING:   llvm.br ^bb3(%[[regarg0]] : !llvm.ptr)
    //  CHECK-LOWERING: ^bb2:  // pred: ^bb0
    //  CHECK-LOWERING:   %[[reg2:[a-z0-9]+]] = llvm.mlir.constant(64 : i64) : i64
    //  CHECK-LOWERING:   %[[reg3:[a-z0-9]+]] = llvm.mlir.constant(16 : i64) : i64
    //  CHECK-LOWERING:   %[[reg4:[a-z0-9]+]] = llvm.call @__reussir_alloc(%[[reg2]], %[[reg3]]) : (i64, i64) -> !llvm.ptr
    //  CHECK-LOWERING:   llvm.br ^bb3(%[[reg4]] : !llvm.ptr)
    //  CHECK-LOWERING: ^bb3(%[[reg5:[a-z0-9]+]]: !llvm.ptr):  // 2 preds: ^bb1, ^bb2
    //  CHECK-LOWERING:   llvm.br ^bb4
    //  CHECK-LOWERING: ^bb4:  // pred: ^bb3
    //  CHECK-LOWERING:   llvm.return %[[reg5]] : !llvm.ptr
    //  CHECK-LOWERING: }
    func.func @token_ensure_expansion(%0: !reussir.nullable<!tk>) -> !tk{
        %1 = reussir.token.ensure (%0 : !reussir.nullable<!tk>) : !tk
        return %1 : !tk
    }
}
