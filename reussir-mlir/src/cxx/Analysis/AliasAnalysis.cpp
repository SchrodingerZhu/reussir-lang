#include "Reussir/Analysis/AliasAnalysis.h"
#include "Reussir/IR/ReussirOps.h"
#include "mlir/Analysis/AliasAnalysis.h"
namespace mlir {
namespace reussir {
mlir::AliasResult AliasAnalysis::alias(mlir::Value lhs, mlir::Value rhs) {
  if (lhs == rhs)
    return mlir::AliasResult::MustAlias;

  if (isa<RcType>(lhs.getType()) && isa<RcType>(rhs.getType())) {
    // are they loaded from a must-alias reference?
    auto lhsLoad = dyn_cast_or_null<LoadOp>(lhs.getDefiningOp());
    auto rhsLoad = dyn_cast_or_null<LoadOp>(rhs.getDefiningOp());
    if (lhsLoad && rhsLoad &&
        alias(lhsLoad.getObject(), rhsLoad.getObject()) ==
            mlir::AliasResult::MustAlias)
      return mlir::AliasResult::MustAlias;
  }

  if (isa<RefType>(lhs.getType()) && isa<RefType>(rhs.getType())) {
    // are they projected from a must-alias reference?
    auto lhsProj = dyn_cast_or_null<ProjOp>(lhs.getDefiningOp());
    auto rhsProj = dyn_cast_or_null<ProjOp>(rhs.getDefiningOp());
    if (lhsProj && rhsProj && lhsProj.getIndex() == rhsProj.getIndex() &&
        alias(lhsProj.getObject(), rhsProj.getObject()) ==
            mlir::AliasResult::MustAlias)
      return mlir::AliasResult::MustAlias;
    // are they borrowed from a must-alias reference?
    auto lhsBorrow = dyn_cast_or_null<RcBorrowOp>(lhs.getDefiningOp());
    auto rhsBorrow = dyn_cast_or_null<RcBorrowOp>(rhs.getDefiningOp());
    if (lhsBorrow && rhsBorrow &&
        alias(lhsBorrow.getObject(), rhsBorrow.getObject()) ==
            mlir::AliasResult::MustAlias)
      return mlir::AliasResult::MustAlias;
    // are they inspected from a must-alias reference?
    auto lhsInspect = dyn_cast_or_null<UnionInspectOp>(lhs.getDefiningOp());
    auto rhsInspect = dyn_cast_or_null<UnionInspectOp>(rhs.getDefiningOp());
    if (lhsInspect && rhsInspect &&
        alias(lhsInspect.getUnionRef(), rhsInspect.getUnionRef()) ==
            mlir::AliasResult::MustAlias)
      return mlir::AliasResult::MustAlias;
  }

  // We skip possible alias for mref/nullable rc types. they are not
  // subject to reuse analysis. We also ignore possible must-alias across
  // assembled structures.
  return AliasResult::MayAlias;
}
} // namespace reussir
} // namespace mlir
