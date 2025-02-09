// RUN: %reussir-opt %s
!ilist = !reussir.union<"list" incomplete>
!rclist = !reussir.rc<!ilist, nonatomic, nonfreezing>
!cons = !reussir.composite<"list::cons" {i32, !rclist}>
!nil = !reussir.composite<"list::nil" {}>
!list = !reussir.union<"list" {!cons, !nil}>
!reflist = !reussir.ref<!ilist, nonfreezing>
!list_token = !reussir.token<size: 32, alignment: 8>
!nullable = !reussir.nullable<!list_token>
!refcons = !reussir.ref<!cons, nonfreezing>
!refi32 = !reussir.ref<i32, nonfreezing>
!refrc = !reussir.ref<!rclist, nonfreezing>
module @test  attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<f80, dense<128> : vector<2xi64>>, #dlti.dl_entry<i128, dense<128> : vector<2xi64>>, #dlti.dl_entry<i32, dense<32> : vector<2xi64>>, #dlti.dl_entry<f128, dense<128> : vector<2xi64>>, #dlti.dl_entry<f64, dense<64> : vector<2xi64>>, #dlti.dl_entry<f16, dense<16> : vector<2xi64>>, #dlti.dl_entry<i1, dense<8> : vector<2xi64>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi64>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi64>>, #dlti.dl_entry<i8, dense<8> : vector<2xi64>>, #dlti.dl_entry<i16, dense<16> : vector<2xi64>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi64>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi64>>, #dlti.dl_entry<i64, dense<64> : vector<2xi64>>, #dlti.dl_entry<"dlti.stack_alignment", 128 : i64>, #dlti.dl_entry<"dlti.endianness", "little">>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"} {
    func.func @reverse(%list: !rclist, %acc: !rclist) -> !rclist {
        %ref = reussir.rc.borrow %list : !rclist -> !reflist
        %tag = reussir.union.get_tag %ref : !reflist -> index
        %res = scf.index_switch %tag -> !rclist 
        case 0 {
            %inner = reussir.union.inspect %ref [0] : !reflist -> !refcons
            %refhead = reussir.proj %inner [0] : !refcons -> !refi32
            %head = reussir.load %refhead : !refi32 -> i32
            %reftail = reussir.proj %inner [1] : !refcons -> !refrc
            %tail = reussir.load %reftail : !refrc -> !rclist
            reussir.rc.acquire (%tail : !rclist)
            %tk = reussir.rc.release (%list : !rclist) : !nullable
            %cons = reussir.composite.assemble (%head, %acc) : (i32, !rclist) -> !cons
            %next = reussir.union.assemble (0, %cons) : (!cons) -> !list
            %token = reussir.token.alloc : !list_token
            %res = reussir.rc.create value(%next) token(%token) : (!list, !list_token) -> !rclist
            %recusive = func.call @reverse(%tail, %res) : (!rclist, !rclist) -> !rclist
            scf.yield %recusive : !rclist
        }
        case 1 {
            reussir.union.inspect %ref [1] : !reflist
            %x = reussir.rc.release (%list : !rclist) : !nullable
            reussir.token.free (%x : !nullable)
            scf.yield %acc : !rclist
        }
        default {
            %y = reussir.unreachable : !rclist
            scf.yield %y : !rclist
        }
        return %res : !rclist
    }
}
