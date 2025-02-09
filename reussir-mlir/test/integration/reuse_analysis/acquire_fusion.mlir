// RUN: %reussir-opt %s -reussir-expand-control-flow=outline-nested-release=0 -reussir-acquire-release-fusion 2>&1 | %FileCheck %s
!rc = !reussir.rc<i64, nonatomic, nonfreezing>
!refrc = !reussir.ref<!rc, nonfreezing>
!struct = !reussir.composite<{!rc, !rc, !rc}>
!box = !reussir.rc<!struct, nonatomic, nonfreezing>
!ref = !reussir.ref<!struct, nonfreezing>
!tk1 = !reussir.token<size: 32, alignment: 8>
!tk2 = !reussir.token<size: 16, alignment: 8>
module @test attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<f80, dense<128> : vector<2xi64>>, #dlti.dl_entry<i128, dense<128> : vector<2xi64>>, #dlti.dl_entry<i32, dense<32> : vector<2xi64>>, #dlti.dl_entry<f128, dense<128> : vector<2xi64>>, #dlti.dl_entry<f64, dense<64> : vector<2xi64>>, #dlti.dl_entry<f16, dense<16> : vector<2xi64>>, #dlti.dl_entry<i1, dense<8> : vector<2xi64>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi64>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi64>>, #dlti.dl_entry<i8, dense<8> : vector<2xi64>>, #dlti.dl_entry<i16, dense<16> : vector<2xi64>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi64>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi64>>, #dlti.dl_entry<i64, dense<64> : vector<2xi64>>, #dlti.dl_entry<"dlti.stack_alignment", 128 : i64>, #dlti.dl_entry<"dlti.endianness", "little">>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"} {
    // CHECK: reussir.rc.acquire(%{{[a-z0-9]+}} : <i64, nonatomic, nonfreezing>)
    // CHECK-NEXT: reussir.rc.acquire(%{{[a-z0-9]+}} : <i64, nonatomic, nonfreezing>)
    // CHECK-NEXT: %[[REG:[a-z0-9]+]] = reussir.nullable.null : <!reussir.token<size : 32, alignment : 8>>
    // CHECK-NEXT: scf.yield %[[REG]] : !reussir.nullable<!reussir.token<size : 32, alignment : 8>>
    func.func @fusion(%0: !box) -> !rc {
        %ref = reussir.rc.borrow %0 : !box -> !ref
        %proj = reussir.proj %ref [1] : !ref -> !refrc
        %valrc = reussir.load %proj : !refrc -> !rc
        %proj2 = reussir.proj %ref [2] : !ref -> !refrc
        %valrc2 = reussir.load %proj2 : !refrc -> !rc
        reussir.rc.acquire (%valrc2 : !rc)
        reussir.rc.acquire (%valrc : !rc)
        %tk = reussir.rc.release (%0 : !box) : !reussir.nullable<!tk1>
        %tk2 = reussir.rc.release (%valrc2 : !rc) : !reussir.nullable<!tk2>
        reussir.token.free (%tk : !reussir.nullable<!tk1>)
        reussir.token.free (%tk2 : !reussir.nullable<!tk2>)
        return %valrc : !rc
    }
}
