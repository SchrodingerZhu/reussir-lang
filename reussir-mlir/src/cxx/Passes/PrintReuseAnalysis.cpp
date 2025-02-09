#include "Reussir/Analysis/ReuseAnalysis.h"
#include "Reussir/Interfaces/ReussirCompositeLayoutInterface.h"
#include "Reussir/Passes.h"
#include "mlir/Analysis/AliasAnalysis.h"
#include "mlir/Analysis/DataFlow/ConstantPropagationAnalysis.h"
#include "mlir/Analysis/DataFlow/DeadCodeAnalysis.h"
#include "mlir/Analysis/DataLayoutAnalysis.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Dominance.h"
#include "mlir/Interfaces/DataLayoutInterfaces.h"
#include <memory>

namespace mlir {
namespace reussir {

struct ReussirPrintReuseAnalysisPass
    : public ReussirPrintReuseAnalysisBase<ReussirPrintReuseAnalysisPass> {
  using ReussirPrintReuseAnalysisBase::ReussirPrintReuseAnalysisBase;
  void runOnOperation() override final;
};

void ReussirPrintReuseAnalysisPass::runOnOperation() {
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
  getOperation().walk([&](Operation *op) {
    const auto *lattice =
        solver.lookupState<dataflow::reussir::ReuseLattice>(op);
    if (!lattice)
      return;
    std::string buffer;
    llvm::raw_string_ostream os(buffer);
    lattice->print(os);
    llvm::SmallVector<Attribute> types;
    op->setAttr("reuse-analysis", StringAttr::get(op->getContext(), buffer));
  });
}

// NOLINTNEXTLINE(misc-use-internal-linkage)
std::unique_ptr<Pass> createReussirPrintReuseAnalysisPass() {
  return std::make_unique<ReussirPrintReuseAnalysisPass>();
}
} // namespace reussir
} // namespace mlir
