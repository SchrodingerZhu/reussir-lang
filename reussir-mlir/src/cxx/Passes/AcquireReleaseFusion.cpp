#include "Reussir/Analysis/AliasAnalysis.h"
#include "Reussir/Common.h"
#include "Reussir/IR/ReussirOps.h"
#include "Reussir/IR/ReussirTypes.h"
#include "Reussir/Passes.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Dominance.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Value.h"
#include "mlir/Support/LLVM.h"
#include "llvm/ADT/SmallVector.h"
#include <algorithm>
#include <memory>

namespace mlir {

namespace REUSSIR_DECL_SCOPE {

struct ReussirAcquireReleaseFusionPass
    : public ReussirAcquireReleaseFusionBase<ReussirAcquireReleaseFusionPass> {
  using ReussirAcquireReleaseFusionBase::ReussirAcquireReleaseFusionBase;
  void runOnOperation() override final {
    // for now, let's only handle acquire-release in the same block
    auto func = getOperation();
    mlir::AliasAnalysis aliasAnalysis(func);
    mlir::PostDominanceInfo postDomInfo(func);
    aliasAnalysis.addAnalysisImplementation<mlir::reussir::AliasAnalysis>({});
    llvm::DenseSet<Operation *> toErase;
    llvm::DenseSet<RcReleaseOp> toRewrite;
    llvm::SmallVector<std::pair<RcAcquireOp, Block *>> toMove;
    func->walk([&](RcAcquireOp acquire) {
      // find the scf::IfOp following the acquire
      auto *next = acquire->getNextNode();
      while (next) {
        if (auto ifOp = dyn_cast<scf::IfOp>(next)) {
          // check if this is a branch expanded from release operation
          if (ifOp->hasAttr(RELEASE)) {
            auto &dropRegion = ifOp.getThenRegion();
            // find the fusion target
            RcReleaseOp fusionTarget{};
            dropRegion.walk([&](RcReleaseOp release) {
              if (!fusionTarget && !toErase.contains(release) &&
                  !toRewrite.contains(release) &&
                  // there can be branches in the dropRegion, we can only fuse
                  // the operation if the release post-dominates the region.
                  // The acquire naturally dominates the release.
                  postDomInfo.postDominates(release->getBlock(),
                                            &dropRegion.front()) &&
                  aliasAnalysis.alias(release.getRcPtr(), acquire.getRcPtr()) ==
                      AliasResult::MustAlias)
                fusionTarget = release;
            });
            if (fusionTarget) {
              // fuse the acquire and release
              toErase.insert(fusionTarget);
              toMove.push_back({acquire, &ifOp.getElseRegion().front()});
              break;
            }
          }
        }
        // if there is somehow a release in between, we can fuse them
        if (auto release = dyn_cast<RcReleaseOp>(next)) {
          if (!toErase.contains(release) && !toRewrite.contains(release) &&
              aliasAnalysis.alias(release.getRcPtr(), acquire.getRcPtr()) ==
                  AliasResult::MustAlias) {
            if (release->getUses().empty())
              toErase.insert(release);
            else
              toRewrite.insert(release);
            toErase.insert(acquire);
            break;
          }
        }
        // avoid analysis across function calls
        if (isa<func::CallOp>(next))
          break;
        next = next->getNextNode();
      }
    });
    IRRewriter rewriter(&getContext());
    for (auto *op : toErase)
      rewriter.eraseOp(op);
    for (auto op : toRewrite) {
      rewriter.setInsertionPoint(op);
      rewriter.replaceOpWithNewOp<NullableNullOp>(op, op.getResultTypes());
    }
    for (auto [acquire, block] : toMove) {
      rewriter.setInsertionPointToStart(block);
      rewriter.clone(*acquire.getOperation());
      rewriter.eraseOp(acquire);
    }
  };
};
} // namespace REUSSIR_DECL_SCOPE

namespace reussir {
std::unique_ptr<Pass> createReussirAcquireReleaseFusionPass() {
  return std::make_unique<ReussirAcquireReleaseFusionPass>();
}
} // namespace reussir
} // namespace mlir
