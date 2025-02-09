// RUN: %reussir-opt %s \
// RUN:   -reussir-gen-freezable-vtable \
// RUN:   -reussir-expand-control-flow=outline-nested-release=1 \
// RUN:   -convert-scf-to-cf \
// RUN:   -canonicalize \
// RUN:   -convert-reussir-to-llvm \
// RUN:   -convert-to-llvm -reconcile-unrealized-casts | %mlir-translate -mlir-to-llvmir | %opt -O3 -S | %FileCheck %s
!urc = !reussir.rc<i64, nonatomic, unfrozen>
!frc = !reussir.rc<i64, nonatomic, frozen>
!mref = !reussir.mref<i64, nonatomic>
!rci64 = !reussir.rc<i64, nonatomic, nonfreezing>
!struct = !reussir.composite<"test" {!mref, !mref, !rci64}>
!usrc = !reussir.rc<!struct, nonatomic, unfrozen>
!fsrc = !reussir.rc<!struct, nonatomic, frozen>
!nullable = !reussir.nullable<!urc>
!i64token = !reussir.token<size: 32, alignment: 8>
!stoken = !reussir.token<size: 48, alignment: 8>
// CHECK-DAG: @"i64::$fvtable" = linkonce_odr dso_local constant { ptr, ptr, i64, i64, i64 } { ptr null, ptr null, i64 32, i64 8, i64 24 }, align 8
// CHECK-DAG: @"test::$fvtable" = linkonce_odr dso_local constant { ptr, ptr, i64, i64, i64 } { ptr @"test::$drop", ptr @"test::$scan", i64 48, i64 8, i64 24 }, align 8
// CHECK: define linkonce_odr void @"test::$drop"(ptr %0) {
// CHECK: define linkonce_odr void @"test::$scan"(ptr %0, ptr %1, ptr %2) {
module @test attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<f80, dense<128> : vector<2xi64>>, #dlti.dl_entry<i128, dense<128> : vector<2xi64>>, #dlti.dl_entry<i32, dense<32> : vector<2xi64>>, #dlti.dl_entry<f128, dense<128> : vector<2xi64>>, #dlti.dl_entry<f64, dense<64> : vector<2xi64>>, #dlti.dl_entry<f16, dense<16> : vector<2xi64>>, #dlti.dl_entry<i1, dense<8> : vector<2xi64>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi64>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi64>>, #dlti.dl_entry<i8, dense<8> : vector<2xi64>>, #dlti.dl_entry<i16, dense<16> : vector<2xi64>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi64>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi64>>, #dlti.dl_entry<i64, dense<64> : vector<2xi64>>, #dlti.dl_entry<"dlti.stack_alignment", 128 : i64>, #dlti.dl_entry<"dlti.endianness", "little">>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"} {
    func.func @rc_release(%x : i64, %y : i64, %z: i64) -> !fsrc {
        %1 = reussir.region.run {
          ^bb(%ctx : !reussir.region_ctx):
             %null = reussir.nullable.null : !nullable
             // CHECK: tail call align 8 ptr @__reussir_alloc(i64 32, i64 8)
             %token = reussir.token.alloc : !i64token 
             %urc_ = reussir.rc.create value(%x) token(%token) region(%ctx) : (i64, !i64token, !reussir.region_ctx) -> !urc
             %urc = reussir.nullable.nonnull (%urc_ : !urc) : !nullable
             // CHECK: tail call align 8 ptr @__reussir_alloc(i64 16, i64 8)
             %tk = reussir.token.alloc : !reussir.token<size: 16, alignment: 8>
             %i64rc = reussir.rc.create value(%z) token(%tk) : (i64, !reussir.token<size: 16, alignment: 8>) -> !rci64
             %struct = reussir.composite.assemble (%urc, %null, %i64rc) : (!nullable, !nullable, !rci64) -> !struct
             // CHECK: tail call align 8 ptr @__reussir_alloc(i64 48, i64 8)
             %stoken = reussir.token.alloc : !stoken
             %usrc = reussir.rc.create value(%struct) token(%stoken) region(%ctx) : (!struct, !stoken, !reussir.region_ctx) -> !usrc
             %borrowed = reussir.rc.borrow %usrc : !usrc -> !reussir.ref<!struct, unfrozen>
             %proj = reussir.proj %borrowed[1] : !reussir.ref<!struct, unfrozen> -> !reussir.ref<!mref, unfrozen>
             // CHECK: tail call align 8 ptr @__reussir_alloc(i64 32, i64 8)
             %token1 = reussir.token.alloc : !i64token 
             %urc2_ = reussir.rc.create value(%y) token(%token1) region(%ctx) : (i64, !i64token, !reussir.region_ctx) -> !urc
             %urc2 = reussir.nullable.nonnull (%urc2_ : !urc) : !nullable
             reussir.mref.assign %urc2 to %proj : !nullable, !reussir.ref<!mref, unfrozen>
             // CHECK: tail call void @__reussir_freeze
             %fsrc = reussir.rc.freeze (%usrc : !usrc) : !fsrc
             // CHECK-NEXT: call void @__reussir_clean_up_region
             reussir.region.yield %fsrc : !fsrc
        } : !fsrc
        return %1 : !fsrc
    }
}
