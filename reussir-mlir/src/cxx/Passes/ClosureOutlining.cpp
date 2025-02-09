#include "Reussir/Common.h"
#include "Reussir/IR/ReussirOps.h"
#include "Reussir/IR/ReussirOpsEnums.h"
#include "Reussir/IR/ReussirTypes.h"
#include "Reussir/Interfaces/ReussirCompositeLayoutInterface.h"
#include "Reussir/Passes.h"
#include "mlir/Conversion/FuncToLLVM/ConvertFuncToLLVM.h"
#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMAttrs.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMTypes.h"
#include "mlir/IR/Block.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Value.h"
#include "mlir/IR/ValueRange.h"
#include "mlir/Interfaces/DataLayoutInterfaces.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/Support/ErrorHandling.h"
#include <algorithm>
#include <memory>
#include <optional>

namespace mlir {

namespace REUSSIR_DECL_SCOPE {

class ReussirClosureOutliningPattern : public OpRewritePattern<ClosureNewOp> {
  llvm::DenseMap<llvm::StringRef, size_t> &assignedLambda;
  DataLayout dataLayout;

  void emitOutlinedFunc(ClosureNewOp op, PatternRewriter &rewriter,
                        func::FuncOp lambdaFuncOp, RefType argPackRefTy,
                        FreezingKindAttr nonfreezing) const {
    mlir::OpBuilder::InsertionGuard guard(rewriter);
    Block *funcOpBlock = lambdaFuncOp.addEntryBlock();
    rewriter.setInsertionPointToStart(funcOpBlock);
    llvm::SmallVector<Value> args;
    std::transform(
        op->getRegion(0).args_begin(), op->getRegion(0).args_end(),
        std::back_inserter(args), [&](const mlir::BlockArgument &innerArg) {
          mlir::Value ref = rewriter.create<ProjOp>(
              op->getLoc(),
              RefType::get(getContext(), innerArg.getType(), nonfreezing),
              funcOpBlock->getArgument(0),
              rewriter.getIndexAttr(innerArg.getArgNumber()));
          mlir::Value loaded =
              rewriter.create<LoadOp>(op->getLoc(), innerArg.getType(), ref);
          return loaded;
        });
    rewriter.inlineBlockBefore(&op->getRegion(0).front(), funcOpBlock,
                               funcOpBlock->end(), args);
    rewriter.inlineRegionBefore(op->getRegion(0),
                                lambdaFuncOp.getFunctionBody(),
                                lambdaFuncOp.getFunctionBody().end());
    for (auto &block : lambdaFuncOp.getFunctionBody().getBlocks())
      if (auto yield = dyn_cast_or_null<ClosureYieldOp>(block.getTerminator()))
        rewriter.replaceOpWithNewOp<func::ReturnOp>(yield,
                                                    yield->getOperands());
  }

  void emitOutlinedDrop(ClosureNewOp op, PatternRewriter &rewriter,
                        func::FuncOp dropFunc, RefType argPackRefTy,
                        FreezingKindAttr nonfreezing) const {
    mlir::OpBuilder::InsertionGuard guard(rewriter);
    auto *entryBlock = dropFunc.addEntryBlock();
    auto argPackTy = cast<CompositeType>(argPackRefTy.getPointee());
    auto argPackLayout = argPackTy.getCompositeLayout(dataLayout);
    auto refToArgPack = entryBlock->getArgument(0);
    auto appliedByteOffset = entryBlock->getArgument(1);
    rewriter.setInsertionPointToStart(entryBlock);
    for (auto [idx, ty] : llvm::enumerate(argPackTy.getInnerTypes())) {
      if (ty.isIntOrIndexOrFloat())
        continue;
      auto offset = argPackLayout.getField(idx).byteOffset;
      auto offsetVal =
          rewriter.create<arith::ConstantIndexOp>(op->getLoc(), offset);
      auto greaterThan = rewriter.create<arith::CmpIOp>(
          op->getLoc(), arith::CmpIPredicate::ugt, appliedByteOffset,
          offsetVal);
      rewriter.create<mlir::scf::IfOp>(
          op->getLoc(), greaterThan, [&](OpBuilder &builder, Location loc) {
            auto proj = builder.create<ProjOp>(
                loc, RefType::get(getContext(), ty, nonfreezing), refToArgPack,
                builder.getIndexAttr(idx));
            builder.create<DestroyOp>(loc, proj, IntegerAttr{});
            builder.create<mlir::scf::YieldOp>(loc);
          });
    }
    rewriter.create<func::ReturnOp>(op->getLoc());
  }

public:
  template <typename... Args>
  ReussirClosureOutliningPattern(
      llvm::DenseMap<llvm::StringRef, size_t> &assignedLambda,
      DataLayout dataLayout, Args &&...args)
      : OpRewritePattern(std::forward<Args>(args)...),
        assignedLambda(assignedLambda), dataLayout(dataLayout) {}

  LogicalResult matchAndRewrite(ClosureNewOp op,
                                PatternRewriter &rewriter) const final {
    auto funcOp = dyn_cast<func::FuncOp>(op->getParentOp());
    if (!funcOp)
      return failure();
    auto moduleOp = dyn_cast<ModuleOp>(funcOp->getParentOp());
    if (!moduleOp)
      return failure();
    llvm::SmallVector<mlir::Type> argTypes;
    std::transform(op->getRegion(0).args_begin(), op->getRegion(0).args_end(),
                   std::back_inserter(argTypes),
                   [](BlockArgument arg) { return arg.getType(); });
    auto argPackTy = CompositeType::get(getContext(), argTypes);
    auto nonfreezing =
        FreezingKindAttr::get(getContext(), FreezingKind::nonfreezing);
    auto argPackRefTy = RefType::get(getContext(), argPackTy, nonfreezing);
    // block number from the function
    auto lambdaNumber = assignedLambda[funcOp.getName()]++;
    std::string lambdaName =
        (funcOp.getName() + llvm::Twine("$$lambda") + llvm::Twine(lambdaNumber))
            .str();
    std::string lambdaFuncName = (lambdaName + llvm::Twine("$$func")).str();
    std::string lambdaCloneName = (lambdaName + llvm::Twine("$$clone")).str();
    std::string lambdaDropName = (lambdaName + llvm::Twine("$$drop")).str();
    std::string lambdaVtableName = (lambdaName + llvm::Twine("$$vtable")).str();
    auto refOfClosureTy =
        RefType::get(getContext(), op.getResult().getType(), nonfreezing);
    auto funcTy = FunctionType::get(getContext(), argPackRefTy,
                                    op.getClosureType().getOutputType());
    auto cloneTy = FunctionType::get(getContext(),
                                     {refOfClosureTy, rewriter.getIndexType()},
                                     op.getResult().getType());
    auto dropTy = FunctionType::get(
        getContext(), {argPackRefTy, rewriter.getIndexType()}, {});
    rewriter.setInsertionPoint(funcOp);
    auto lambdaFuncOp =
        rewriter.create<func::FuncOp>(op->getLoc(), lambdaFuncName, funcTy);
    auto lambdaCloneOp =
        rewriter.create<func::FuncOp>(op->getLoc(), lambdaCloneName, cloneTy);
    auto lambdaDropOp =
        rewriter.create<func::FuncOp>(op->getLoc(), lambdaDropName, dropTy);
    lambdaFuncOp.setPrivate();
    lambdaCloneOp.setPrivate();
    lambdaDropOp.setPrivate();
    emitOutlinedFunc(op, rewriter, lambdaFuncOp, argPackRefTy, nonfreezing);
    emitOutlinedDrop(op, rewriter, lambdaDropOp, argPackRefTy, nonfreezing);
    // TODO: emit clone and drop functions
    rewriter.create<ClosureVTableOp>(op->getLoc(), lambdaVtableName,
                                     op.getClosureType(), lambdaFuncName,
                                     lambdaCloneName, lambdaDropName);
    rewriter.setInsertionPoint(op);
    mlir::TypedValue<TokenType> argpack;
    if (argTypes.size() > 0) {
      CompositeLayout layout = argPackTy.getCompositeLayout(dataLayout);
      auto tokenTy = TokenType::get(getContext(), layout.getAlignment().value(),
                                    layout.getSize());
      argpack = rewriter.create<TokenAllocOp>(op->getLoc(), tokenTy);
    }
    rewriter.replaceOpWithNewOp<ClosureAssembleOp>(
        op, op.getType(), SymbolRefAttr::get(getContext(), lambdaVtableName),
        argpack);
    return LogicalResult::success();
  };
};

struct ReussirClosureOutliningPass
    : public ReussirClosureOutliningBase<ReussirClosureOutliningPass> {
  using ReussirClosureOutliningBase::ReussirClosureOutliningBase;
  void runOnOperation() override final;
};

void ReussirClosureOutliningPass::runOnOperation() {
  auto module = getOperation();
  DataLayout dataLayout(module);
  // Collect rewrite patterns.
  RewritePatternSet patterns(&getContext());
  llvm::DenseMap<llvm::StringRef, size_t> assignedLambda;
  patterns.add<ReussirClosureOutliningPattern>(assignedLambda, dataLayout,
                                               &getContext());

  // Collect operations to be considered by the pass.
  SmallVector<Operation *, 16> ops;
  getOperation()->walk([&](ClosureNewOp op) { ops.push_back(op); });

  // Configure rewrite to ignore new ops created during the pass.
  GreedyRewriteConfig config;
  config.strictMode = GreedyRewriteStrictness::AnyOp;

  // Apply patterns.
  if (failed(applyOpPatternsAndFold(ops, std::move(patterns), config)))
    signalPassFailure();
}

} // namespace REUSSIR_DECL_SCOPE

namespace reussir {
std::unique_ptr<Pass> createReussirClosureOutliningPass() {
  return std::make_unique<ReussirClosureOutliningPass>();
}
} // namespace reussir
} // namespace mlir
