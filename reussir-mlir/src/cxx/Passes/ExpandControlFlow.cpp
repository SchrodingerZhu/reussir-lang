#include "Reussir/Common.h"
#include "Reussir/IR/ReussirOps.h"
#include "Reussir/IR/ReussirTypes.h"
#include "Reussir/Interfaces/ReussirCompositeLayoutInterface.h"
#include "Reussir/Passes.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMAttrs.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/IR/Value.h"
#include "mlir/Interfaces/DataLayoutInterfaces.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include <algorithm>
#include <cstdint>
#include <memory>
#include <numeric>

namespace mlir {

namespace REUSSIR_DECL_SCOPE {

#define GEN_PASS_DEF_REUSSIREXPANDCONTROLFLOW
#include "Reussir/Passes.h.inc"

class RegionRunExpansionPattern : public OpRewritePattern<RegionRunOp> {
public:
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(RegionRunOp op,
                                PatternRewriter &rewriter) const final {
    auto &region = op.getBody();
    auto execute = rewriter.create<scf::ExecuteRegionOp>(op->getLoc(),
                                                         op->getResultTypes());
    Value regionCtx;
    {
      OpBuilder::InsertionGuard guard(rewriter);
      auto argument = region.front().getArgument(0);
      rewriter.setInsertionPointToStart(&region.front());
      regionCtx = rewriter.create<RegionCreateOp>(op->getLoc());
      rewriter.replaceAllUsesWith(argument, regionCtx);
    }
    region.front().eraseArgument(0);
    region.walk([&](RegionYieldOp yield) {
      OpBuilder::InsertionGuard guard(rewriter);
      rewriter.setInsertionPoint(yield);
      rewriter.create<RegionCleanUpOp>(yield->getLoc(), regionCtx);
      rewriter.replaceOpWithNewOp<scf::YieldOp>(yield, yield.getValue());
    });
    rewriter.inlineRegionBefore(region, execute.getRegion(),
                                execute.getRegion().end());
    rewriter.replaceOp(op, execute);
    return LogicalResult::success();
  }
};

class TokenEnsureExpansionPattern : public OpRewritePattern<TokenEnsureOp> {
public:
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(TokenEnsureOp op,
                                PatternRewriter &rewriter) const final {
    auto tokenTy =
        cast<TokenType>(op.getNullableToken().getType().getPointer());
    auto isNonNull =
        rewriter.create<NullableCheckOp>(op->getLoc(), op.getNullableToken());
    rewriter.replaceOpWithNewOp<scf::IfOp>(
        op, isNonNull,
        [&](OpBuilder &builder, Location loc) {
          mlir::Value ensured = builder.create<NullableCoerceOp>(
              loc, tokenTy, op.getNullableToken());
          if (tokenTy.getSize() != op.getType().getSize()) {
            assert(tokenTy.getAlignment() == op.getType().getAlignment());
            ensured = builder.create<TokenReallocOp>(loc, tokenTy, ensured);
          }
          builder.create<scf::YieldOp>(loc, ensured);
        },
        [&](OpBuilder &builder, Location loc) {
          auto allocated = builder.create<TokenAllocOp>(loc, tokenTy);
          builder.create<scf::YieldOp>(loc, allocated->getResults());
        });
    return LogicalResult::success();
  }
};

class TokenFreeExpansionPattern : public OpRewritePattern<TokenFreeOp> {
public:
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(TokenFreeOp op,
                                PatternRewriter &rewriter) const final {
    auto nullableTy = dyn_cast<NullableType>(op.getToken().getType());
    if (!nullableTy)
      return LogicalResult::failure();
    auto tokenTy = cast<TokenType>(nullableTy.getPointer());
    auto isNonNull =
        rewriter.create<NullableCheckOp>(op->getLoc(), op.getToken());
    rewriter.replaceOpWithNewOp<scf::IfOp>(
        op, isNonNull,
        [&](OpBuilder &builder, Location loc) {
          auto coerced =
              builder.create<NullableCoerceOp>(loc, tokenTy, op.getToken());
          builder.create<TokenFreeOp>(loc, coerced);
          builder.create<scf::YieldOp>(loc);
        },
        nullptr);
    return LogicalResult::success();
  }
};

template <bool OUTLINE_NESTED>
class RcReleaseExpansionPattern : public OpRewritePattern<RcReleaseOp> {
  void emitCallToOutlinedRelease(RcReleaseOp op,
                                 PatternRewriter &rewriter) const {
    std::string uniqueName;
    llvm::raw_string_ostream os(uniqueName);
    formatMangledNameTo(op.getRcPtr().getType(), os);
    os << "::release";
    SymbolTable symbolTable(op->getParentOfType<ModuleOp>());
    auto func =
        dyn_cast_if_present<func::FuncOp>(symbolTable.lookup(uniqueName));
    if (!func) {
      OpBuilder::InsertionGuard guard(rewriter);
      rewriter.setInsertionPoint(op->getParentOfType<func::FuncOp>());
      auto funcTy = FunctionType::get(rewriter.getContext(),
                                      op.getRcPtr().getType(), TypeRange{});
      func = rewriter.create<func::FuncOp>(op->getLoc(), uniqueName, funcTy);
      func.setSymVisibility("private");
      func->setAttr("llvm.linkage", LLVM::LinkageAttr::get(
                                        rewriter.getContext(),
                                        LLVM::linkage::Linkage::LinkonceODR));
      auto *entry = func.addEntryBlock();
      rewriter.setInsertionPointToStart(entry);
      auto token =
          rewriter.create<RcReleaseOp>(op->getLoc(), op.getToken().getType(),
                                       entry->getArgument(0), op.getTagAttr());
      rewriter.create<TokenFreeOp>(op->getLoc(), token.getToken());
      rewriter.create<func::ReturnOp>(op->getLoc());
    }
    rewriter.create<func::CallOp>(op->getLoc(), func, op.getRcPtr());
  }

public:
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(RcReleaseOp op,
                                PatternRewriter &rewriter) const final {
    RcType rcTy = op.getRcPtr().getType();
    // we only need to expand for nonfreezing RC pointers
    if (rcTy.getFreezingKind().getValue() != FreezingKind::nonfreezing)
      return LogicalResult::failure();
    if (op->hasAttr(NESTED)) {
      if (OUTLINE_NESTED) {
        emitCallToOutlinedRelease(op, rewriter);
        rewriter.eraseOp(op);
        return LogicalResult::success();
      }
      return LogicalResult::failure();
    }
    auto decreaseOp =
        rewriter.create<RcDecreaseOp>(op->getLoc(), op.getRcPtr());
    scf::IfOp branch = rewriter.replaceOpWithNewOp<scf::IfOp>(
        op, decreaseOp,
        [&](OpBuilder &builder, Location loc) {
          auto refTy = RefType::get(getContext(), rcTy.getPointee(),
                                    rcTy.getFreezingKind());
          auto borrowed = builder.create<RcBorrowOp>(loc, refTy, op.getRcPtr());
          builder.create<DestroyOp>(loc, borrowed, op.getTagAttr());
          auto resTy = cast<NullableType>(op.getToken().getType());
          auto token = builder.create<RcTokenizeOp>(loc, resTy.getPointer(),
                                                    op.getRcPtr());
          auto nonnull = builder.create<NullableNonNullOp>(loc, resTy, token);
          builder.create<scf::YieldOp>(loc, nonnull->getResults());
        },
        [&](OpBuilder &builder, Location loc) {
          auto null = builder.create<NullableNullOp>(loc, op.getResultTypes());
          builder.create<scf::YieldOp>(loc, null->getResults());
        });
    branch->setAttr(RELEASE, rewriter.getUnitAttr());
    return LogicalResult::success();
  }
};

class DestroyExpansionPattern : public OpRewritePattern<DestroyOp> {
  CompositeLayoutCache &cache;
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
  bool generateDestroy(PatternRewriter &rewriter, Value target,
                       IntegerAttr tag) const {
    auto refTy = cast<RefType>(target.getType());
    // primitive types
    if (!needDrop(refTy.getPointee()))
      return true;

    // rc pointer, apply RcReleaseOp
    if (auto rcTy = dyn_cast<RcType>(refTy.getPointee())) {
      auto loaded = rewriter.create<LoadOp>(target.getLoc(), rcTy, target);
      if (rcTy.getFreezingKind().getValue() == FreezingKind::nonfreezing) {
        auto rcBoxTy =
            RcBoxType::get(getContext(), rcTy.getPointee(),
                           rcTy.getAtomicKind(), rcTy.getFreezingKind());
        auto size = cache.get(rcBoxTy).getSize();
        auto align = cache.get(rcBoxTy).getAlignment();
        auto tokenTy = TokenType::get(getContext(), align.value(), size);
        auto nullableTy = NullableType::get(getContext(), tokenTy);
        auto release = rewriter.create<RcReleaseOp>(target.getLoc(), nullableTy,
                                                    loaded, nullptr);
        // avoid nested release to be expanded
        release->setAttr(NESTED, rewriter.getUnitAttr());
      } else
        rewriter.create<RcReleaseOp>(target.getLoc(), Type{}, loaded, nullptr);

      return true;
    }

    // for composite type, iterate through the fields and generate the destroy
    // recursively
    if (auto compositeTy = dyn_cast<CompositeType>(refTy.getPointee())) {
      if (!needDrop(compositeTy))
        return true;
      for (auto [i, ty] : llvm::enumerate(compositeTy.getInnerTypes())) {
        if (ty.isIntOrIndexOrFloat())
          continue;
        auto fieldRefTy =
            RefType::get(getContext(), ty, refTy.getFreezingKind());
        auto field = rewriter.create<ProjOp>(target.getLoc(), fieldRefTy,
                                             target, rewriter.getIndexAttr(i));
        generateDestroy(rewriter, field, {});
      }
      return true;
    }

    // for union type, expand it to index_switch
    if (auto unionTy = dyn_cast<UnionType>(refTy.getPointee())) {
      if (tag) {
        auto ty = unionTy.getInnerTypes()[tag.getAPSInt().getZExtValue()];
        auto fieldRef = RefType::get(getContext(), ty, refTy.getFreezingKind());
        auto ref = rewriter.create<UnionInspectOp>(target.getLoc(), fieldRef,
                                                   target, tag);
        generateDestroy(rewriter, ref.getResult(), {});
        return true;
      }
      auto getTag = rewriter.create<UnionGetTagOp>(target.getLoc(), target);
      llvm::SmallVector<int64_t> cases;
      cases.resize(unionTy.getInnerTypes().size());
      std::iota(cases.begin(), cases.end(), 0);
      auto switchOp = rewriter.create<scf::IndexSwitchOp>(
          target.getLoc(), TypeRange{}, getTag, cases,
          unionTy.getInnerTypes().size());
      for (auto [idx, ty] : llvm::enumerate(unionTy.getInnerTypes())) {
        OpBuilder::InsertionGuard guard(rewriter);
        auto &region = switchOp.getCaseRegions()[idx];
        auto *blk = rewriter.createBlock(&region);
        rewriter.setInsertionPointToStart(blk);
        if (needDrop(ty)) {
          auto fieldRef =
              RefType::get(getContext(), ty, refTy.getFreezingKind());
          auto ref = rewriter.create<UnionInspectOp>(
              target.getLoc(), fieldRef, target, rewriter.getIndexAttr(idx));
          generateDestroy(rewriter, ref.getResult(), {});
        }
        rewriter.create<scf::YieldOp>(target.getLoc());
      }
      // insert unreachable for the default case
      {
        OpBuilder::InsertionGuard guard(rewriter);
        auto &region = switchOp.getDefaultRegion();
        auto *blk = rewriter.createBlock(&region);
        rewriter.setInsertionPointToStart(blk);
        rewriter.create<UnreachableOp>(target.getLoc(), Type{});
        rewriter.create<scf::YieldOp>(target.getLoc());
      }
      return true;
    }

    // for other types: closure, opaque, etc. left a destroy placeholder to be
    // lowered in LLVM lowering pass.
    return false;
  }

public:
  using OpRewritePattern::OpRewritePattern;

  template <typename... Args>
  DestroyExpansionPattern(CompositeLayoutCache &cache, Args &&...args)
      : OpRewritePattern(std::forward<Args>(args)...), cache(cache) {}

  LogicalResult matchAndRewrite(DestroyOp op,
                                PatternRewriter &rewriter) const final {
    if (generateDestroy(rewriter, op.getObject(), op.getTagAttr())) {
      rewriter.eraseOp(op);
      return LogicalResult::success();
    }
    return LogicalResult::failure();
  }
};

struct ReussirExpandControlFlowPass
    : public impl::ReussirExpandControlFlowBase<ReussirExpandControlFlowPass> {
  using ReussirExpandControlFlowBase::ReussirExpandControlFlowBase;
  void runOnOperation() override final;
};

void ReussirExpandControlFlowPass::runOnOperation() {
  auto module = getOperation();
  DataLayout dataLayout(module);
  CompositeLayoutCache cache(dataLayout);
  // Collect rewrite patterns.
  RewritePatternSet patterns(&getContext());
  patterns.add<DestroyExpansionPattern>(cache, &getContext());
  patterns.add<TokenEnsureExpansionPattern, TokenFreeExpansionPattern,
               RegionRunExpansionPattern>(&getContext());

  if (outlineNestedRelease)
    patterns.add<RcReleaseExpansionPattern<true>>(&getContext());
  else
    patterns.add<RcReleaseExpansionPattern<false>>(&getContext());

  // Configure rewrite to ignore new ops created during the pass.
  GreedyRewriteConfig config;
  config.strictMode = GreedyRewriteStrictness::ExistingAndNewOps;

  // // Apply patterns.
  if (failed(applyPatternsAndFoldGreedily(module, std::move(patterns), config)))
    signalPassFailure();
}

} // namespace REUSSIR_DECL_SCOPE

namespace reussir {
std::unique_ptr<Pass> createReussirExpandControlFlowPass() {
  return std::make_unique<ReussirExpandControlFlowPass>();
}
std::unique_ptr<Pass> createReussirExpandControlFlowPass(
    const ReussirExpandControlFlowOptions &options) {
  return std::make_unique<ReussirExpandControlFlowPass>(options);
};
} // namespace reussir
} // namespace mlir
