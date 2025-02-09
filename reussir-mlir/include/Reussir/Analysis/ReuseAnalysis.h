#pragma once
#include "Reussir/Analysis/AliasAnalysis.h"
#include "Reussir/IR/ReussirOps.h"
#include "Reussir/IR/ReussirTypes.h"
#include "Reussir/Interfaces/ReussirCompositeLayoutInterface.h"
#include "mlir/Analysis/DataFlow/DenseAnalysis.h"
#include "mlir/Analysis/DataFlowFramework.h"
#include "mlir/Analysis/DataLayoutAnalysis.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Dominance.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/IR/Value.h"
#include "mlir/IR/ValueRange.h"
#include "mlir/Interfaces/DataLayoutInterfaces.h"
#include "mlir/Support/LLVM.h"
#include "llvm/ADT/STLExtras.h"
#include <memory>

namespace mlir::dataflow {
namespace reussir {
using namespace mlir::reussir;
class TokenHeuristic {
private:
  mlir::AliasAnalysis &aliasAnalysis;

private:
  long similarity(Value token, RcCreateOp op) const;

  static inline constexpr size_t MIN_ALLOC_STEP_SIZE = 2 * sizeof(void *);
  static inline constexpr size_t MIN_ALLOC_STEP_BITS =
      __builtin_ctz(MIN_ALLOC_STEP_SIZE);
  static inline constexpr size_t INTERMEDIATE_BITS = 2;
  static size_t toExpMand(size_t value);

  bool possiblyInplaceReallocable(size_t alignment, size_t oldSize,
                                  size_t newSize) const;

public:
  TokenHeuristic(mlir::AliasAnalysis &aliasAnalysis);

  ssize_t operator()(TokenAllocOp op, Value token) const;
};

class ReuseLattice : public AbstractDenseLattice {
  Value reuseToken{};
  llvm::DenseSet<Value> freeToken{};
  llvm::DenseSet<Value> aliveToken{};

public:
  using AbstractDenseLattice::AbstractDenseLattice;
  ChangeResult join(const AbstractDenseLattice &rhs) override final;
  void print(llvm::raw_ostream &os) const override final;
  Value getReuseToken() const { return reuseToken; }
  const llvm::DenseSet<Value> &getFreeToken() const { return freeToken; }
  const llvm::DenseSet<Value> &getAliveToken() const { return aliveToken; }
  ChangeResult setNewState(Value reuseToken, llvm::DenseSet<Value> freeToken,
                           llvm::DenseSet<Value> aliveToken);
};

class ReuseAnalysis : public DenseForwardDataFlowAnalysis<ReuseLattice> {
private:
  TokenHeuristic tokenHeuristic;
  DominanceInfo &domInfo;
#if LLVM_VERSION_MAJOR < 20
  using RetType = void;
#else
  using RetType = LogicalResult;
#endif
  void customVisitBlock(Block *block);
  void customVisitRegionBranchOperation(ProgramPoint point,
                                        RegionBranchOpInterface branch,
                                        AbstractDenseLattice *after);

public:
  ReuseAnalysis(DataFlowSolver &solver, mlir::AliasAnalysis &aliasAnalysis,
                DominanceInfo &domInfo);

  RetType visitOperation(Operation *op, const ReuseLattice &before,
                         ReuseLattice *after) override final;
  RetType success();
  void setToEntryState(ReuseLattice *lattice) override final;
  LogicalResult visit(ProgramPoint point) override final;
  RetType processOperation(Operation *op) override final;
};
} // namespace reussir
} // namespace mlir::dataflow
