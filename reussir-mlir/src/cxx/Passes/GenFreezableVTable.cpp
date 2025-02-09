#include "Reussir/Analysis/AliasAnalysis.h"
#include "Reussir/Common.h"
#include "Reussir/IR/ReussirOps.h"
#include "Reussir/IR/ReussirOpsEnums.h"
#include "Reussir/IR/ReussirTypes.h"
#include "Reussir/Interfaces/ReussirCompositeLayoutInterface.h"
#include "Reussir/Interfaces/ReussirMangleInterface.h"
#include "Reussir/Passes.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Dominance.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Types.h"
#include "mlir/IR/Value.h"
#include "mlir/IR/ValueRange.h"
#include "mlir/Interfaces/DataLayoutInterfaces.h"
#include "mlir/Support/LLVM.h"
#include "llvm/ADT/SmallVector.h"
#include <memory>
#include <numeric>

namespace mlir {

namespace REUSSIR_DECL_SCOPE {

struct ReussirGenFreezableVTablePass
    : public ReussirGenFreezableVTableBase<ReussirGenFreezableVTablePass> {
  using ReussirGenFreezableVTableBase::ReussirGenFreezableVTableBase;
  void runOnOperation() override final;
  bool needDrop(Type type) const {
    if (type.isIntOrIndexOrFloat() || isa<MRefType>(type))
      return false;
    if (auto compositeTy = dyn_cast<CompositeType>(type)) {
      return std::any_of(compositeTy.getInnerTypes().begin(),
                         compositeTy.getInnerTypes().end(),
                         [&](Type ty) { return needDrop(ty); });
    }
    if (auto arrayTy = dyn_cast<ArrayType>(type))
      return needDrop(arrayTy.getElementType());
    if (auto unionTy = dyn_cast<UnionType>(type))
      return std::any_of(unionTy.getInnerTypes().begin(),
                         unionTy.getInnerTypes().end(),
                         [&](Type ty) { return needDrop(ty); });
    return true;
  }
  bool needScan(Type type) const {
    if (type.isIntOrIndexOrFloat() || isa<RcType>(type))
      return false;
    if (auto compositeTy = dyn_cast<CompositeType>(type)) {
      return std::any_of(compositeTy.getInnerTypes().begin(),
                         compositeTy.getInnerTypes().end(),
                         [&](Type ty) { return needScan(ty); });
    }
    if (auto arrayTy = dyn_cast<ArrayType>(type))
      return needScan(arrayTy.getElementType());
    if (auto unionTy = dyn_cast<UnionType>(type))
      return std::any_of(unionTy.getInnerTypes().begin(),
                         unionTy.getInnerTypes().end(),
                         [&](Type ty) { return needScan(ty); });
    return true;
  }

  void emitScan(OpBuilder &builder, Value action, Value ctx, Value current) {
    auto refTy = cast<RefType>(current.getType());
    if (!needScan(refTy.getPointee()))
      return;

    if (auto mref = dyn_cast<MRefType>(refTy.getPointee())) {
      auto rcTy = RcType::get(&getContext(), mref.getPointee(),
                              mref.getAtomicKind(), refTy.getFreezingKind());
      auto nullableTy = NullableType::get(&getContext(), rcTy);
      auto loaded =
          builder.create<LoadOp>(current.getLoc(), nullableTy, current);
      auto isNonnull =
          builder.create<NullableCheckOp>(current.getLoc(), loaded);
      builder.create<scf::IfOp>(
          current.getLoc(), isNonnull, [&](OpBuilder &builder, Location loc) {
            auto coerced = builder.create<NullableCoerceOp>(loc, rcTy, loaded);
            auto opaque = builder.create<RcAsPtrOp>(current.getLoc(), coerced);
            builder.create<func::CallIndirectOp>(current.getLoc(), action,
                                                 ValueRange{opaque, ctx});
            builder.create<scf::YieldOp>(current.getLoc());
          });
      return;
    }

    if (auto compositeTy = dyn_cast<CompositeType>(refTy.getPointee())) {
      for (auto [i, ty] : llvm::enumerate(compositeTy.getInnerTypes())) {
        if (!needScan(ty))
          continue;
        auto fieldRefTy =
            RefType::get(&getContext(), ty, refTy.getFreezingKind());
        auto field = builder.create<ProjOp>(current.getLoc(), fieldRefTy,
                                            current, builder.getIndexAttr(i));
        emitScan(builder, action, ctx, field);
      }
      return;
    }

    if (auto unionTy = dyn_cast<UnionType>(refTy.getPointee())) {
      auto getTag = builder.create<UnionGetTagOp>(current.getLoc(), current);
      llvm::SmallVector<int64_t> cases;
      cases.resize(unionTy.getInnerTypes().size());
      std::iota(cases.begin(), cases.end(), 0);
      auto switchOp = builder.create<scf::IndexSwitchOp>(
          current.getLoc(), TypeRange{}, getTag, cases,
          unionTy.getInnerTypes().size());
      for (auto [idx, ty] : llvm::enumerate(unionTy.getInnerTypes())) {
        OpBuilder::InsertionGuard guard(builder);
        auto &region = switchOp.getCaseRegions()[idx];
        auto *blk = builder.createBlock(&region);
        builder.setInsertionPointToStart(blk);
        if (needScan(ty)) {
          auto fieldRef =
              RefType::get(&getContext(), ty, refTy.getFreezingKind());
          auto ref = builder.create<UnionInspectOp>(
              current.getLoc(), fieldRef, current, builder.getIndexAttr(idx));
          emitScan(builder, action, ctx, ref.getResult());
        }
        builder.create<scf::YieldOp>(current.getLoc());
      }
      // insert unreachable for the default case
      {
        OpBuilder::InsertionGuard guard(builder);
        auto &region = switchOp.getDefaultRegion();
        auto *blk = builder.createBlock(&region);
        builder.setInsertionPointToStart(blk);
        builder.create<UnreachableOp>(current.getLoc(), Type{});
        builder.create<scf::YieldOp>(current.getLoc());
      }
      return;
    }

    llvm_unreachable("TODO: array and other types");
  }
};

void ReussirGenFreezableVTablePass::runOnOperation() {
  auto module = getOperation();
  DataLayout layout(module);
  CompositeLayoutCache cache(layout);

  DenseSet<Type> unfrozenTypes;
  module->walk([&](Operation *op) {
    for (auto ty : op->getResultTypes())
      if (auto rcTy = dyn_cast<RcType>(ty))
        if (rcTy.getFreezingKind().getValue() != FreezingKind::nonfreezing)
          unfrozenTypes.insert(rcTy.getPointee());
  });

  if (unfrozenTypes.empty())
    return;

  OpBuilder builder(module->getContext());
  builder.setInsertionPointToStart(module.getBody());
  auto opaquePtrTy = PtrType::get(module->getContext());
  auto actionFnTy = builder.getFunctionType({opaquePtrTy, opaquePtrTy}, {});
  for (auto ty : unfrozenTypes) {
    std::string mangledName;
    llvm::raw_string_ostream os(mangledName);
    formatMangledNameTo(ty, os);
    auto refTy = RefType::get(
        module->getContext(), ty,
        FreezingKindAttr::get(&getContext(), FreezingKind::unfrozen));
    func::FuncOp dropFn{}, scanFn{};
    auto dropName = mangledName + "::$drop";
    if (needDrop(ty)) {
      dropFn = builder.create<func::FuncOp>(
          module.getLoc(), dropName, builder.getFunctionType({refTy}, {}));
      dropFn.setPrivate();
    }
    auto scanName = mangledName + "::$scan";
    if (needScan(ty)) {
      scanFn = builder.create<func::FuncOp>(
          module.getLoc(), scanName,
          builder.getFunctionType({refTy, actionFnTy, opaquePtrTy}, {}));
      scanFn.setPrivate();
    }
    auto rcBoxTy = RcBoxType::get(
        module->getContext(), ty,
        AtomicKindAttr::get(&getContext(), AtomicKind::nonatomic),
        FreezingKindAttr::get(&getContext(), FreezingKind::unfrozen));
    auto compositeLayout = cache.get(rcBoxTy);
    builder.create<FreezableVTableOp>(
        module->getLoc(),
        FlatSymbolRefAttr::get(&getContext(), mangledName + "::$fvtable"),
        dropFn ? FlatSymbolRefAttr::get(&getContext(), dropName)
               : FlatSymbolRefAttr{},
        scanFn ? FlatSymbolRefAttr::get(&getContext(), scanName)
               : FlatSymbolRefAttr{},
        builder.getIndexAttr(compositeLayout.getSize()),
        builder.getIndexAttr(compositeLayout.getAlignment().value()),
        builder.getIndexAttr(compositeLayout.getField(3).byteOffset));
    if (dropFn) {
      OpBuilder::InsertionGuard guard(builder);
      auto *entry = dropFn.addEntryBlock();
      builder.setInsertionPointToStart(entry);
      builder.create<DestroyOp>(module.getLoc(), entry->getArgument(0),
                                nullptr);
      builder.create<func::ReturnOp>(module.getLoc());
      dropFn->setAttr("llvm.linkage",
                      LLVM::LinkageAttr::get(
                          &getContext(), LLVM::linkage::Linkage::LinkonceODR));
    }
    if (scanFn) {
      OpBuilder::InsertionGuard guard(builder);
      auto *entry = scanFn.addEntryBlock();
      builder.setInsertionPointToStart(entry);
      auto ref = entry->getArgument(0);
      auto action = entry->getArgument(1);
      auto ctx = entry->getArgument(2);
      emitScan(builder, action, ctx, ref);
      builder.create<func::ReturnOp>(module.getLoc());
      scanFn->setAttr("llvm.linkage",
                      LLVM::LinkageAttr::get(
                          &getContext(), LLVM::linkage::Linkage::LinkonceODR));
    }
  }
}
} // namespace REUSSIR_DECL_SCOPE

namespace reussir {
std::unique_ptr<Pass> createReussirGenFreezableVTablePass() {
  return std::make_unique<ReussirGenFreezableVTablePass>();
}
} // namespace reussir
} // namespace mlir
