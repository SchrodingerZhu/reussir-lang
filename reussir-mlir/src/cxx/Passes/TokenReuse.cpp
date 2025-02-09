#include "Reussir/Analysis/ReuseAnalysis.h"
#include "Reussir/IR/ReussirOps.h"
#include "Reussir/Interfaces/ReussirCompositeLayoutInterface.h"
#include "Reussir/Passes.h"
#include "mlir/Analysis/AliasAnalysis.h"
#include "mlir/Analysis/DataFlow/ConstantPropagationAnalysis.h"
#include "mlir/Analysis/DataFlow/DeadCodeAnalysis.h"
#include "mlir/Analysis/DataFlowFramework.h"
#include "mlir/Analysis/DataLayoutAnalysis.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Dominance.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Interfaces/ControlFlowInterfaces.h"
#include "mlir/Interfaces/DataLayoutInterfaces.h"
#include <memory>

namespace mlir {
namespace reussir {

struct ReussirTokenReusePass
    : public ReussirTokenReuseBase<ReussirTokenReusePass> {
  using ReussirTokenReuseBase::ReussirTokenReuseBase;
  void runOnOperation() override final;
};

void ReussirTokenReusePass::runOnOperation() {
  auto module = getOperation();
  mlir::AliasAnalysis aliasAnalysis(module);
  DominanceInfo dominanceInfo(module);
  aliasAnalysis.addAnalysisImplementation<mlir::reussir::AliasAnalysis>({});
  auto config = DataFlowConfig().setInterprocedural(false);
  DataFlowSolver solver(config);
  solver.load<dataflow::DeadCodeAnalysis>();
  solver.load<dataflow::reussir::ReuseAnalysis>(aliasAnalysis, dominanceInfo);
  solver.load<dataflow::SparseConstantPropagation>();
  if (failed(solver.initializeAndRun(getOperation()))) {
    emitError(getOperation()->getLoc(), "dataflow solver failed");
    return signalPassFailure();
  }
  // first walk all operations:
  // - if an operation reuses a token, add it to its token slot.
  // - if an operation frees tokens, insert token free operations right before
  // it
  IRRewriter rewriter(&getContext());
  getOperation().walk([&](Operation *op) {
    const auto *lattice =
        solver.lookupState<dataflow::reussir::ReuseLattice>(op);
    if (!lattice)
      return;

    if (auto alloc = dyn_cast<TokenAllocOp>(op)) {
      if (lattice->getReuseToken()) {
        rewriter.setInsertionPoint(op);
        rewriter.replaceOpWithNewOp<TokenEnsureOp>(alloc, alloc.getType(),
                                                   lattice->getReuseToken());
      }
    }

    for (auto toFree : lattice->getFreeToken()) {
      rewriter.setInsertionPoint(op);
      rewriter.create<TokenFreeOp>(op->getLoc(), toFree);
    }

    if (auto terminator = dyn_cast<RegionBranchTerminatorOpInterface>(op)) {
      auto *parent = terminator->getParentOp();
      auto *parentLattice =
          solver.lookupState<dataflow::reussir::ReuseLattice>(parent);
      // free all locally alive tokens that are not alive at parentLattice
      llvm::DenseSet<Value> toFree;
      for (auto token : lattice->getAliveToken())
        if (!parentLattice->getAliveToken().contains(token))
          toFree.insert(token);
      for (auto token : toFree) {
        rewriter.setInsertionPoint(op);
        rewriter.create<TokenFreeOp>(token.getLoc(), token);
      }
    } else if (op->hasTrait<OpTrait::IsTerminator>() &&
               op->getBlock()->hasNoSuccessors()) {
      // For normal terminator, free all alive tokens
      for (auto token : lattice->getAliveToken()) {
        rewriter.setInsertionPoint(op);
        rewriter.create<TokenFreeOp>(token.getLoc(), token);
      }
    }
  });
  // Walk all blocks, if a token alive at the end of the block but it is
  // not alive at the beginning of a successor block, insert token free at the
  // beginning of the successor block. It is garanteed that the token alive must
  // dominate all successors.
  getOperation().walk([&](Block *block) {
    auto *lattice = solver.lookupState<dataflow::reussir::ReuseLattice>(
        ProgramPoint(block));
    if (!lattice)
      return;
    llvm::DenseSet<Value> toFree;
    for (Block::pred_iterator begin = block->pred_begin(),
                              end = block->pred_end();
         begin != end; ++begin) {
      auto *pred = *begin;
      auto *predLattice = solver.lookupState<dataflow::reussir::ReuseLattice>(
          pred->getTerminator());
      if (!predLattice)
        continue;
      for (auto token : predLattice->getAliveToken())
        if (!lattice->getAliveToken().contains(token))
          toFree.insert(token);
    }
    for (auto token : toFree) {
      rewriter.setInsertionPoint(block, block->begin());
      rewriter.create<TokenFreeOp>(token.getLoc(), token);
    }
  });
}

// NOLINTNEXTLINE(misc-use-internal-linkage)
std::unique_ptr<Pass> createReussirTokenReusePass() {
  return std::make_unique<ReussirTokenReusePass>();
}
} // namespace reussir
} // namespace mlir
