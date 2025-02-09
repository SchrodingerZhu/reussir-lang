#pragma once

#include "Reussir/Common.h"
#include "Reussir/IR/ReussirOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"

namespace mlir {
namespace reussir {
std::unique_ptr<Pass> createConvertReussirToLLVMPass();
std::unique_ptr<Pass> createReussirClosureOutliningPass();
std::unique_ptr<Pass> createReussirExpandControlFlowPass();
std::unique_ptr<Pass> createReussirExpandControlFlowPass(
    const struct ReussirExpandControlFlowOptions &options);
std::unique_ptr<Pass> createReussirAcquireReleaseFusionPass();
std::unique_ptr<Pass> createReussirInferUnionTagPass();
std::unique_ptr<Pass> createReussirPrintReuseAnalysisPass();
std::unique_ptr<Pass> createReussirTokenReusePass();
std::unique_ptr<Pass> createReussirGenFreezableVTablePass();

#define GEN_PASS_CLASSES
#define GEN_PASS_REGISTRATION
#define GEN_PASS_DECL
#include "Reussir/Passes.h.inc"

inline constexpr llvm::StringLiteral NESTED = "reussir.nested";
inline constexpr llvm::StringLiteral RELEASE = "reussir.release";

} // namespace reussir
} // namespace mlir
