#include "Reussir/Analysis/ReuseAnalysis.h"

#include "Reussir/IR/ReussirOps.h"
#include "Reussir/IR/ReussirTypes.h"
#include "mlir/Analysis/DataFlow/DeadCodeAnalysis.h"
#include "mlir/Analysis/DataFlowFramework.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Block.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Operation.h"
#include "mlir/Interfaces/ControlFlowInterfaces.h"

namespace mlir::dataflow {
namespace reussir {

void ReuseLattice::print(llvm::raw_ostream &os) const {
  {
    os << "{";
    if (this->reuseToken) {
      os << "reuse: ";
      reuseToken.printAsOperand(os, OpPrintingFlags{});
      os << ", ";
    }
    os << "free : [";
    llvm::interleaveComma(this->freeToken, os, [&](Value token) {
      token.printAsOperand(os, OpPrintingFlags{});
    });
    os << "], alive: [";
    llvm::interleaveComma(aliveToken, os, [&](Value token) {
      token.printAsOperand(os, OpPrintingFlags{});
    });
    os << "]}";
  }
}

ChangeResult ReuseLattice::join(const AbstractDenseLattice &rhs) {
  llvm_unreachable("should not be called, the join operation is customized");
}

void ReuseAnalysis::setToEntryState(ReuseLattice *lattice) {
  propagateIfChanged(lattice, lattice->setNewState({}, {}, {}));
}

ReuseAnalysis::RetType ReuseAnalysis::visitOperation(Operation *op,
                                                     const ReuseLattice &before,
                                                     ReuseLattice *after) {
  Value reuseToken{};
  llvm::DenseSet<Value> freeToken{};
  llvm::DenseSet<Value> aliveToken = before.getAliveToken();
  do {
    if (auto release = dyn_cast<RcReleaseOp>(op)) {
      if (release.getToken() && release.getToken().use_empty())
        aliveToken.insert(release.getToken());
      break;
    }

    if (auto alloc = dyn_cast<TokenAllocOp>(op)) {
      Value reuseCandidate{};
      long currentHeuristic = -10;
      for (auto alive : aliveToken) {
        auto heuristic = tokenHeuristic(alloc, alive);
        if (!reuseCandidate || heuristic > currentHeuristic) {
          reuseCandidate = alive;
          currentHeuristic = heuristic;
        }
      }
      // reusable
      if (currentHeuristic >= -1)
        reuseToken = reuseCandidate;
      aliveToken.erase(reuseToken);
      break;
    }
  } while (false);

  if (op->getBlock()->getTerminator() == op) {
    // if any alive token does not dominate one of the successors, free it.
    llvm::SmallVector<Value> toFree;
    for (auto token : aliveToken) {
      for (auto *succ : op->getBlock()->getSuccessors()) {
        if (!domInfo.dominates(token.getParentBlock(), succ)) {
          toFree.push_back(token);
          break;
        }
      }
    }
    for (auto token : toFree) {
      freeToken.insert(token);
      aliveToken.erase(token);
    }
  }

  propagateIfChanged(after,
                     after->setNewState(reuseToken, freeToken, aliveToken));

#if LLVM_VERSION_MAJOR < 20
  return;
#else
  return LogicalResult::success();
#endif
}

ssize_t TokenHeuristic::operator()(TokenAllocOp op, Value token) const {
  auto nullableTy = cast<NullableType>(token.getType());
  auto tokenTy = cast<TokenType>(nullableTy.getPointer());
  auto targetTy = op.getType();
  if (tokenTy.getAlignment() != targetTy.getAlignment())
    return -2;
  if (tokenTy.getSize() != targetTy.getSize())
    return possiblyInplaceReallocable(tokenTy.getAlignment(), tokenTy.getSize(),
                                      targetTy.getSize())
               ? -1
               : -2;
  if (op->hasOneUse()) {
    if (auto rcCreateOp = dyn_cast<RcCreateOp>(*op->user_begin()))
      return similarity(token, rcCreateOp);
  }
  return 0;
}

long TokenHeuristic::similarity(Value token, RcCreateOp op) const {
  auto rcReleaseOp = dyn_cast_or_null<RcReleaseOp>(token.getDefiningOp());
  if (!rcReleaseOp)
    return 0;

  auto assembleOp = dyn_cast_or_null<CompositeAssembleOp>(
      rcReleaseOp.getOperand().getDefiningOp());
  if (!assembleOp)
    return 0;
  long score = 0;
  for (auto field : assembleOp.getFields()) {
    auto *defOp = field.getDefiningOp();
    while (defOp && defOp->getNumResults() && defOp->getNumOperands() &&
           !isa<RcType>(defOp->getResultTypes()[0])) {
      if (isa_and_nonnull<LoadOp, ProjOp, UnionInspectOp, RcBorrowOp>(defOp)) {
        defOp = defOp->getOperand(0).getDefiningOp();
        continue;
      }
    }
    if (defOp && defOp->getNumResults() && defOp->getNumOperands() &&
        isa<RcType>(defOp->getResultTypes()[0]) &&
        aliasAnalysis.alias(rcReleaseOp.getRcPtr(), defOp->getResult(0)) ==
            AliasResult::MustAlias)
      score++;
  }
  return score;
}
size_t TokenHeuristic::toExpMand(size_t value) {
  auto oneAtBit = [](size_t bit) { return 1 << bit; };
  constexpr size_t LEADING_BIT =
      oneAtBit(INTERMEDIATE_BITS + MIN_ALLOC_STEP_BITS) >> 1;
  constexpr size_t MANTISSA_MASK = oneAtBit(INTERMEDIATE_BITS) - 1;
  constexpr size_t BITS = sizeof(size_t) * CHAR_BIT;

  value = value - 1;

  size_t e = BITS - INTERMEDIATE_BITS - MIN_ALLOC_STEP_BITS -
             __builtin_clz(value | LEADING_BIT);
  size_t b = (e == 0) ? 0 : 1;
  size_t m = (value >> (MIN_ALLOC_STEP_BITS + e - b)) & MANTISSA_MASK;

  return (e << INTERMEDIATE_BITS) + m;
}
bool TokenHeuristic::possiblyInplaceReallocable(size_t alignment,
                                                size_t oldSize,
                                                size_t newSize) const {
  auto alignedSize = [](size_t alignment, size_t size) {
    return ((alignment - 1) | (size - 1)) + 1;
  };

  constexpr size_t GB = 1024 * 1024 * 1024;
  auto oldAlignedSize = alignedSize(alignment, oldSize);
  auto newAlignedSize = alignedSize(alignment, newSize);
  if (oldAlignedSize >= GB || newAlignedSize >= GB)
    return false;
  auto oldExpMand = toExpMand(oldAlignedSize >> MIN_ALLOC_STEP_BITS);
  auto newExpMand = toExpMand(newAlignedSize >> MIN_ALLOC_STEP_BITS);
  return newExpMand == oldExpMand;
}
ReuseAnalysis::ReuseAnalysis(DataFlowSolver &solver,
                             mlir::AliasAnalysis &aliasAnalysis,
                             DominanceInfo &domInfo)
    : DenseForwardDataFlowAnalysis(solver), tokenHeuristic(aliasAnalysis),
      domInfo(domInfo) {}
TokenHeuristic::TokenHeuristic(mlir::AliasAnalysis &aliasAnalysis)
    : aliasAnalysis(aliasAnalysis) {}

LogicalResult ReuseAnalysis::visit(ProgramPoint point) {
  if (auto *op = llvm::dyn_cast_if_present<Operation *>(point)) {
#if LLVM_VERSION_MAJOR < 20
    processOperation(op);
    return LogicalResult::success();
#else
    return processOperation(op);
#endif
  }
  customVisitBlock(point.get<Block *>());
  return LogicalResult::success();
}

void ReuseAnalysis::customVisitBlock(Block *block) {
  // If the block is not executable, bail out.
  if (!getOrCreateFor<Executable>(block, block)->isLive())
    return;

  // Get the dense lattice to update.
  ReuseLattice *after = getLattice(block);

  // The dense lattices of entry blocks are set by region control-flow or the
  // callgraph.
  if (block->isEntryBlock()) {
    // Check if this block is the entry block of a callable region.
    auto callable = dyn_cast<CallableOpInterface>(block->getParentOp());
    if (callable && callable.getCallableRegion() == block->getParent()) {
      const auto *callsites = getOrCreateFor<PredecessorState>(block, callable);
      // If not all callsites are known, conservatively mark all lattices as
      // having reached their pessimistic fixpoints. Do the same if
      // interprocedural analysis is not enabled.
      if (!callsites->allPredecessorsKnown() ||
          !getSolverConfig().isInterprocedural())
        return setToEntryState(after);
      for (Operation *callsite : callsites->getKnownPredecessors()) {
        // Get the dense lattice before the callsite.
        const ReuseLattice *before;
        if (Operation *prev = callsite->getPrevNode())
          before =
              static_cast<const ReuseLattice *>(getLatticeFor(block, prev));
        else
          before = static_cast<const ReuseLattice *>(
              getLatticeFor(block, callsite->getBlock()));

        visitCallControlFlowTransfer(cast<CallOpInterface>(callsite),
                                     CallControlFlowAction::EnterCallee,
                                     *before, after);
      }
      return;
    }

    // Check if we can reason about the control-flow.
    if (auto branch = dyn_cast<RegionBranchOpInterface>(block->getParentOp()))
      return customVisitRegionBranchOperation(block, branch, after);

    // Otherwise, we can't reason about the data-flow.
    return setToEntryState(after);
  }

  // Join the state with the state after the block's predecessors.
  if (block->pred_begin() == block->pred_end())
    propagateIfChanged(after, after->setNewState({}, {}, {}));

  bool initialized = false;
  llvm::DenseSet<Value> aliveToken;
  for (Block::pred_iterator it = block->pred_begin(), e = block->pred_end();
       it != e; ++it) {
    // Skip control edges that aren't executable.
    Block *predecessor = *it;
#if LLVM_VERSION_MAJOR < 20
    if (!getOrCreateFor<Executable>(
             block, getProgramPoint<CFGEdge>(predecessor, block))
             ->isLive())
      continue;
#else
    if (!getOrCreateFor<Executable>(
             block, getLatticeAnchor<CFGEdge>(predecessor, block))
             ->isLive())
      continue;
#endif

    // intersect the state from the predecessor's terminator.
    const auto &before = static_cast<const ReuseLattice &>(
        *getLatticeFor(block, predecessor->getTerminator()));
    if (!initialized) {
      aliveToken = before.getAliveToken();
      initialized = true;
    } else {
      llvm::SmallVector<Value> toErase;
      for (auto token : aliveToken)
        if (!before.getAliveToken().count(token))
          toErase.push_back(token);
      for (auto token : toErase)
        aliveToken.erase(token);
    }
  }
  propagateIfChanged(after, after->setNewState({}, {}, aliveToken));
};

void ReuseAnalysis::customVisitRegionBranchOperation(
    ProgramPoint point, RegionBranchOpInterface branch,
    AbstractDenseLattice *after) {
  // Get the terminator predecessors.
  const auto *predecessors = getOrCreateFor<PredecessorState>(point, point);
  assert(predecessors->allPredecessorsKnown() &&
         "unexpected unresolved region successors");
  bool initialized = false;
  llvm::DenseSet<Value> aliveToken;
  for (Operation *op : predecessors->getKnownPredecessors()) {
    const AbstractDenseLattice *before;
    // If the predecessor is the parent, get the state before the parent.
    if (op == branch) {
      if (Operation *prev = op->getPrevNode())
        before = getLatticeFor(point, prev);
      else
        before = getLatticeFor(point, op->getBlock());

      // Otherwise, get the state after the terminator.
    } else {
      before = getLatticeFor(point, op);
    }

    // Intersect the state from the predecessor.
    const auto &beforeLattice = static_cast<const ReuseLattice &>(*before);
    if (!initialized) {
      aliveToken = beforeLattice.getAliveToken();
      initialized = true;
    } else {
      llvm::SmallVector<Value> toErase;
      for (auto token : aliveToken)
        if (!beforeLattice.getAliveToken().count(token))
          toErase.push_back(token);
      for (auto token : toErase)
        aliveToken.erase(token);
    }
  }
  auto *afterLattice = static_cast<ReuseLattice *>(after);
  propagateIfChanged(afterLattice,
                     afterLattice->setNewState({}, {}, aliveToken));
}

ReuseAnalysis::RetType ReuseAnalysis::processOperation(
    Operation *op) { // If the containing block is not executable, bail out.
  if (!getOrCreateFor<Executable>(op, op->getBlock())->isLive())
    return this->success();

  // Get the dense lattice to update.
  ReuseLattice *after = getLattice(op);

  // Get the dense state before the execution of the op.
  const ReuseLattice *before;
  if (Operation *prev = op->getPrevNode())
    before = static_cast<const ReuseLattice *>(getLatticeFor(op, prev));
  else
    before =
        static_cast<const ReuseLattice *>(getLatticeFor(op, op->getBlock()));

  // If this op implements region control-flow, then control-flow dictates its
  // transfer function.

  if (auto branch = dyn_cast<RegionBranchOpInterface>(op)) {
    customVisitRegionBranchOperation(op, branch, after);
    return this->success();
  }

  // If this is a call operation, free all reachable tokens.
  if (auto call = dyn_cast<CallOpInterface>(op)) {
    propagateIfChanged(after,
                       after->setNewState({}, before->getAliveToken(), {}));
    return this->success();
  }

  // Invoke the operation transfer function.
  return visitOperationImpl(op, *before, after);
}
ChangeResult ReuseLattice::setNewState(Value reuseToken,
                                       llvm::DenseSet<Value> freeToken,
                                       llvm::DenseSet<Value> aliveToken) {
  auto changed = ChangeResult::NoChange;
  if (reuseToken != this->reuseToken) {
    this->reuseToken = std::move(reuseToken);
    changed = ChangeResult::Change;
  }
  if (freeToken != this->freeToken) {
    this->freeToken = std::move(freeToken);
    changed = ChangeResult::Change;
  }
  if (aliveToken != this->aliveToken) {
    this->aliveToken = std::move(aliveToken);
    changed = ChangeResult::Change;
  }
  return changed;
}
ReuseAnalysis::RetType ReuseAnalysis::success() {
#if LLVM_VERSION_MAJOR < 20
  return;
#else
  return LogicalResult::success();
#endif
}
} // namespace reussir
} // namespace mlir::dataflow
