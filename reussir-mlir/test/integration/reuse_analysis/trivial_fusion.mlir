// RUN: %reussir-opt %s 2>&1 -reussir-acquire-release-fusion 
!rc = !reussir.rc<i64, nonatomic, nonfreezing>
!refrc = !reussir.ref<!rc, nonfreezing>
!struct = !reussir.composite<{!rc, !rc, !rc}>
!box = !reussir.rc<!struct, nonatomic, nonfreezing>
!ref = !reussir.ref<!struct, nonfreezing>
!tk1 = !reussir.token<size: 32, alignment: 8>
!tk2 = !reussir.token<size: 16, alignment: 8>
// CHECK-NOT: reussir.rc.acquire
// CHECK-NOT: reussir.rc.release
module @test attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<f80, dense<128> : vector<2xi64>>, #dlti.dl_entry<i128, dense<128> : vector<2xi64>>, #dlti.dl_entry<i32, dense<32> : vector<2xi64>>, #dlti.dl_entry<f128, dense<128> : vector<2xi64>>, #dlti.dl_entry<f64, dense<64> : vector<2xi64>>, #dlti.dl_entry<f16, dense<16> : vector<2xi64>>, #dlti.dl_entry<i1, dense<8> : vector<2xi64>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi64>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi64>>, #dlti.dl_entry<i8, dense<8> : vector<2xi64>>, #dlti.dl_entry<i16, dense<16> : vector<2xi64>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi64>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi64>>, #dlti.dl_entry<i64, dense<64> : vector<2xi64>>, #dlti.dl_entry<"dlti.stack_alignment", 128 : i64>, #dlti.dl_entry<"dlti.endianness", "little">>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"} {
    func.func @fusion(%0: !box) {
        reussir.rc.acquire (%0 : !box)
        %x = reussir.rc.release (%0 : !box) : !reussir.nullable<!tk1>
        return
    }
    func.func @fusion_used(%0: !box) -> !reussir.nullable<!tk1> {
        // CHECK: reussir.nullable.null
        reussir.rc.acquire (%0 : !box)
        %x = reussir.rc.release (%0 : !box) : !reussir.nullable<!tk1>
        return %x : !reussir.nullable<!tk1>
    }
}
