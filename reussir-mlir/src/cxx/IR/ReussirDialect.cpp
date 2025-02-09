#include "Reussir/IR/ReussirDialect.h"
#include "Reussir/Common.h"
#include "Reussir/IR/ReussirOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinDialect.h"
#include "llvm/Support/ErrorHandling.h"

namespace mlir {
namespace REUSSIR_DECL_SCOPE {
void ReussirDialect::initialize() {
  registerTypes();
  registerAttributes();
  addOperations<
#define GET_OP_LIST
#include "Reussir/IR/ReussirOps.cpp.inc"
      >();
}
::mlir::Operation *
ReussirDialect::materializeConstant(::mlir::OpBuilder &builder,
                                    ::mlir::Attribute value, ::mlir::Type type,
                                    ::mlir::Location loc) {
  llvm_unreachable("TODO");
}
} // namespace REUSSIR_DECL_SCOPE
} // namespace mlir

#include "Reussir/IR/ReussirOpsDialect.cpp.inc"
