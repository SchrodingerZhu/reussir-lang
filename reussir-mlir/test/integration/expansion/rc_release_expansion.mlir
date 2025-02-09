// RUN: %reussir-opt %s 2>&1 -reussir-expand-control-flow | %FileCheck %s
// RUN: %reussir-opt %s 2>&1  -reussir-expand-control-flow -convert-scf-to-cf -convert-reussir-to-llvm | %FileCheck %s --check-prefix=CHECK-LOWERING
!tk = !reussir.token<size: 16, alignment: 8> // TODO: how can we check token is compatible with the rcbox? add an additional pass?
!rc = !reussir.rc<i64, nonatomic, nonfreezing> 
!nullable = !reussir.nullable<!tk>
module @test attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<f80, dense<128> : vector<2xi64>>, #dlti.dl_entry<i128, dense<128> : vector<2xi64>>, #dlti.dl_entry<i32, dense<32> : vector<2xi64>>, #dlti.dl_entry<f128, dense<128> : vector<2xi64>>, #dlti.dl_entry<f64, dense<64> : vector<2xi64>>, #dlti.dl_entry<f16, dense<16> : vector<2xi64>>, #dlti.dl_entry<i1, dense<8> : vector<2xi64>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi64>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi64>>, #dlti.dl_entry<i8, dense<8> : vector<2xi64>>, #dlti.dl_entry<i16, dense<16> : vector<2xi64>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi64>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi64>>, #dlti.dl_entry<i64, dense<64> : vector<2xi64>>, #dlti.dl_entry<"dlti.stack_alignment", 128 : i64>, #dlti.dl_entry<"dlti.endianness", "little">>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"} {
// CHECK:   func.func @rc_release_expansion(%[[regarg0:[a-z0-9]+]]: !reussir.rc<i64, nonatomic, nonfreezing>) -> !reussir.nullable<!reussir.token<size : 16, alignment : 8>> {
// CHECK:    %[[reg0:[a-z0-9]+]] = reussir.rc.decrease(%[[regarg0]] : <i64, nonatomic, nonfreezing>) : i1
// CHECK:    %[[reg1:[a-z0-9]+]] = scf.if %[[reg0]] -> (!reussir.nullable<!reussir.token<size : 16, alignment : 8>>) {
// CHECK:      %[[reg2:[a-z0-9]+]] = reussir.rc.borrow %[[regarg0]] : <i64, nonatomic, nonfreezing> -> <i64, nonfreezing>
// CHECK:      %[[reg3:[a-z0-9]+]] = reussir.rc.tokenize %[[regarg0]] : <i64, nonatomic, nonfreezing> -> <size : 16, alignment : 8>
// CHECK:      %[[reg4:[a-z0-9]+]] = reussir.nullable.nonnull(%[[reg3]] : !reussir.token<size : 16, alignment : 8>) : <!reussir.token<size : 16, alignment : 8>>
// CHECK:      scf.yield %[[reg4]] : !reussir.nullable<!reussir.token<size : 16, alignment : 8>>
// CHECK:    } else {
// CHECK:      %[[reg2:[a-z0-9]+]] = reussir.nullable.null : <!reussir.token<size : 16, alignment : 8>>
// CHECK:      scf.yield %[[reg2]] : !reussir.nullable<!reussir.token<size : 16, alignment : 8>>
// CHECK:    }
// CHECK:    return %[[reg1]] : !reussir.nullable<!reussir.token<size : 16, alignment : 8>>
// CHECK:  }

// CHECK-LOWERING: llvm.func @rc_release_expansion(%[[regarg0:[a-z0-9]+]]: !llvm.ptr) -> !llvm.ptr {
// CHECK-LOWERING:     %[[reg0:[a-z0-9]+]] = llvm.mlir.constant(1 : i64) : i64
// CHECK-LOWERING:     %[[reg1:[a-z0-9]+]] = llvm.getelementptr %[[regarg0]][0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(i64, i64)>
// CHECK-LOWERING:     %[[reg2:[a-z0-9]+]] = llvm.load %[[reg1]] : !llvm.ptr -> i64
// CHECK-LOWERING:     %[[reg3:[a-z0-9]+]] = llvm.sub %[[reg2]], %[[reg0]] : i64
// CHECK-LOWERING:     llvm.store %[[reg3]], %[[reg1]] : i64, !llvm.ptr
// CHECK-LOWERING:     %[[reg4:[a-z0-9]+]] = llvm.mlir.constant(1 : i64) : i64
// CHECK-LOWERING:     %[[reg5:[a-z0-9]+]] = llvm.icmp "eq" %[[reg2]], %[[reg4]] : i64
// CHECK-LOWERING:     llvm.cond_br %[[reg5]], ^bb1, ^bb2
// CHECK-LOWERING:   ^bb1:  // pred: ^bb0
// CHECK-LOWERING:     %[[reg6:[a-z0-9]+]] = llvm.getelementptr %[[regarg0]][0, 1] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(i64, i64)>
// CHECK-LOWERING:     llvm.br ^bb3(%[[regarg0]] : !llvm.ptr)
// CHECK-LOWERING:   ^bb2:  // pred: ^bb0
// CHECK-LOWERING:     %[[reg7:[a-z0-9]+]] = llvm.mlir.zero : !llvm.ptr
// CHECK-LOWERING:     llvm.br ^bb3(%[[reg7]] : !llvm.ptr)
// CHECK-LOWERING:   ^bb3(%[[reg8:[a-z0-9]+]]: !llvm.ptr):  // 2 preds: ^bb1, ^bb2
// CHECK-LOWERING:     llvm.br ^bb4
// CHECK-LOWERING:   ^bb4:  // pred: ^bb3
// CHECK-LOWERING:     llvm.return %[[reg8]] : !llvm.ptr
// CHECK-LOWERING: }
    func.func @rc_release_expansion(%0: !rc) -> !nullable {
        %a = reussir.rc.release (%0 : !rc) : !nullable
        return %a : !nullable
    }
}
