#pragma once
#include "mlir/Analysis/AliasAnalysis.h"
#include "mlir/IR/Value.h"
namespace mlir {
namespace reussir {
struct AliasAnalysis {
  mlir::AliasResult alias(mlir::Value lhs, mlir::Value rhs);
  mlir::ModRefResult getModRef(mlir::Operation *, mlir::Value) {
    // we reply on the default analysis based on memory effect for now.
    return ModRefResult::getModAndRef();
  }
};
} // namespace reussir
} // namespace mlir
