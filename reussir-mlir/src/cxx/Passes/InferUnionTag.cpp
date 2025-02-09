#include "Reussir/Analysis/AliasAnalysis.h"
#include "Reussir/Common.h"
#include "Reussir/IR/ReussirOps.h"
#include "Reussir/Passes.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Dominance.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Value.h"
#include "mlir/Support/LLVM.h"
#include "llvm/ADT/SmallVector.h"
#include <memory>

namespace mlir {

namespace REUSSIR_DECL_SCOPE {

struct ReussirInferUnionTagPass
    : public ReussirInferUnionTagBase<ReussirInferUnionTagPass> {
  using ReussirInferUnionTagBase::ReussirInferUnionTagBase;
  void runOnOperation() override final;
};

void ReussirInferUnionTagPass::runOnOperation() {
  auto func = getOperation();

  // get the dominance information
  auto &domInfo = getAnalysis<DominanceInfo>();

  mlir::AliasAnalysis aliasAnalysis(getOperation());
  aliasAnalysis.addAnalysisImplementation(::mlir::reussir::AliasAnalysis());

  // Collect operations to be considered by the pass.
  SmallVector<UnionInspectOp> inspectOps;
  SmallVector<RcReleaseOp> releaseOps;
  func->walk([&](Operation *op) {
    if (auto inspect = dyn_cast<UnionInspectOp>(op))
      inspectOps.push_back(inspect);
    else if (auto release = dyn_cast<RcReleaseOp>(op))
      releaseOps.push_back(release);
  });

  // apply the changes
  for (auto release : releaseOps)
    for (auto inspect : inspectOps)
      if (auto borrow = dyn_cast_or_null<RcBorrowOp>(
              inspect.getUnionRef().getDefiningOp()))
        if (aliasAnalysis.alias(release.getRcPtr(), borrow.getObject()) ==
                AliasResult::MustAlias &&
            domInfo.dominates(inspect.getOperation(), release.getOperation()))
          release.setTagAttr(inspect.getIndexAttr());
}
} // namespace REUSSIR_DECL_SCOPE

namespace reussir {
std::unique_ptr<Pass> createReussirInferUnionTagPass() {
  return std::make_unique<ReussirInferUnionTagPass>();
}
} // namespace reussir
} // namespace mlir
