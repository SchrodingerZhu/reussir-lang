// RUN: %reussir-opt %s 2>&1 -reussir-expand-control-flow
// TODO: add checks
!istruct = !reussir.composite<"test" incomplete>
!rc = !reussir.rc<!istruct, nonatomic, nonfreezing>
!i64rc = !reussir.rc<i64, nonatomic, nonfreezing>
!nested_composite = !reussir.composite<"a"{!i64rc, !i64rc, !i64rc}>
!nested_union = !reussir.union<"b" {!i64rc, i64, i8, !rc}>
!mstruct = !reussir.composite<"mtest" {!reussir.mref<i64, nonatomic>, i64}>
!frc = !reussir.rc<!mstruct, nonatomic, frozen>
!struct = !reussir.composite<"test" {
    !rc,
    i64,
    !frc,
    !nested_composite,
    !nested_union
}>
!ref = !reussir.ref<!struct, nonfreezing>

module @test attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<f80, dense<128> : vector<2xi64>>, #dlti.dl_entry<i128, dense<128> : vector<2xi64>>, #dlti.dl_entry<i32, dense<32> : vector<2xi64>>, #dlti.dl_entry<f128, dense<128> : vector<2xi64>>, #dlti.dl_entry<f64, dense<64> : vector<2xi64>>, #dlti.dl_entry<f16, dense<16> : vector<2xi64>>, #dlti.dl_entry<i1, dense<8> : vector<2xi64>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi64>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi64>>, #dlti.dl_entry<i8, dense<8> : vector<2xi64>>, #dlti.dl_entry<i16, dense<16> : vector<2xi64>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi64>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi64>>, #dlti.dl_entry<i64, dense<64> : vector<2xi64>>, #dlti.dl_entry<"dlti.stack_alignment", 128 : i64>, #dlti.dl_entry<"dlti.endianness", "little">>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"} {
    func.func @destroy(%0: !ref) {
        reussir.destroy (%0 : !ref)
        return
    }
}
