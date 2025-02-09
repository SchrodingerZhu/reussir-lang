// RUN: %reussir-opt %s
!urc = !reussir.rc<i64, nonatomic, unfrozen>
!frc = !reussir.rc<i64, nonatomic, frozen>
!mref = !reussir.mref<i64, nonatomic>
!struct = !reussir.composite<"test" {!mref, !mref}>
!usrc = !reussir.rc<!struct, nonatomic, unfrozen>
!fsrc = !reussir.rc<!struct, nonatomic, frozen>
!nullable = !reussir.nullable<!urc>
!i64token = !reussir.token<size: 32, alignment: 8>
!stoken = !reussir.token<size: 40, alignment: 8>
module @test attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<f80, dense<128> : vector<2xi64>>, #dlti.dl_entry<i128, dense<128> : vector<2xi64>>, #dlti.dl_entry<i32, dense<32> : vector<2xi64>>, #dlti.dl_entry<f128, dense<128> : vector<2xi64>>, #dlti.dl_entry<f64, dense<64> : vector<2xi64>>, #dlti.dl_entry<f16, dense<16> : vector<2xi64>>, #dlti.dl_entry<i1, dense<8> : vector<2xi64>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi64>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi64>>, #dlti.dl_entry<i8, dense<8> : vector<2xi64>>, #dlti.dl_entry<i16, dense<16> : vector<2xi64>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi64>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi64>>, #dlti.dl_entry<i64, dense<64> : vector<2xi64>>, #dlti.dl_entry<"dlti.stack_alignment", 128 : i64>, #dlti.dl_entry<"dlti.endianness", "little">>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"} {
    func.func @rc_release(%x : i64, %y : i64) -> !fsrc {
        %1 = reussir.region.run {
          ^bb(%ctx : !reussir.region_ctx):
             %null = reussir.nullable.null : !nullable
             %token = reussir.token.alloc : !i64token 
             %urc_ = reussir.rc.create value(%x) token(%token) region(%ctx) : (i64, !i64token, !reussir.region_ctx) -> !urc
             %urc = reussir.nullable.nonnull (%urc_ : !urc) : !nullable
             %struct = reussir.composite.assemble (%urc, %null) : (!nullable, !nullable) -> !struct
             %stoken = reussir.token.alloc : !stoken
             %usrc = reussir.rc.create value(%struct) token(%stoken) region(%ctx) : (!struct, !stoken, !reussir.region_ctx) -> !usrc
             %borrowed = reussir.rc.borrow %usrc : !usrc -> !reussir.ref<!struct, unfrozen>
             %proj = reussir.proj %borrowed[1] : !reussir.ref<!struct, unfrozen> -> !reussir.ref<!mref, unfrozen>
             %token1 = reussir.token.alloc : !i64token 
             %urc2_ = reussir.rc.create value(%y) token(%token1) region(%ctx) : (i64, !i64token, !reussir.region_ctx) -> !urc
             %urc2 = reussir.nullable.nonnull (%urc2_ : !urc) : !nullable
             reussir.mref.assign %urc2 to %proj : !nullable, !reussir.ref<!mref, unfrozen>
             %fsrc = reussir.rc.freeze (%usrc : !usrc) : !fsrc
             reussir.region.yield %fsrc : !fsrc
        } : !fsrc
        return %1 : !fsrc
    }
}
