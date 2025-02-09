#ifndef REUSSIR_PASSES
#define REUSSIR_PASSES

include "mlir/Pass/PassBase.td"

def ConvertReussirToLLVM : Pass<"convert-reussir-to-llvm", "::mlir::ModuleOp"> {
  let summary = "Convert Low-level Reussir Operations to LLVM IR";
  let description = [{
    This pass is serves as the last step in the Reussir pipeline. It converts
    all remaining low-level Reussir operations to LLVM IR.
    ```

  }];
  let constructor = "::mlir::reussir::createConvertReussirToLLVMPass()";
  let dependentDialects = ["::mlir::reussir::ReussirDialect", "::mlir::LLVM::LLVMDialect"];
}

def ReussirClosureOutlining : Pass<"reussir-closure-outlining", "::mlir::ModuleOp"> {
  let summary = "Outlines Reussir Closures";
  let description = [{
    This pass outlines closures and instantiates their virtual tables.
  }];
  let constructor = "::mlir::reussir::createReussirClosureOutliningPass()";
  let dependentDialects = ["::mlir::reussir::ReussirDialect"];
}

def ReussirGenFreezableVTable : Pass<"reussir-gen-freezable-vtable", "::mlir::ModuleOp"> {
  let summary = "Generate vtables for freezable rc objects";
  let description = [{
    This pass generates vtables for freezable rc objects.
  }];
  let constructor = "::mlir::reussir::createReussirGenFreezableVTablePass()";
  let dependentDialects = ["::mlir::reussir::ReussirDialect", "::mlir::scf::SCFDialect"];
}

def ReussirExpandControlFlow : Pass<"reussir-expand-control-flow", "::mlir::ModuleOp"> {
  let summary = "Expand structured control flows for Reussir operations";
  let description = [{
    This pass is used to expand control flows for Reussir operations.
    - reussir.token.ensure
    - reussir.token.free (with nullible token)
    - reussir.rc.release
  }];
    let options = [
    Option<"outlineNestedRelease", "outline-nested-release", "bool", "false", "Outline nested RC release operations">
  ];
  let constructor = "::mlir::reussir::createReussirExpandControlFlowPass()";
  let dependentDialects = ["::mlir::reussir::ReussirDialect", "::mlir::scf::SCFDialect"];
}

def ReussirAcquireReleaseFusion : Pass<"reussir-acquire-release-fusion", "::mlir::func::FuncOp"> {
  let summary = "Fuse Reussir acquire and release Operations";
  let description = [{
    This pass is used to fuse acquire and release operations.
  }];
  let constructor = "::mlir::reussir::createReussirAcquireReleaseFusionPass()";
  let dependentDialects = ["::mlir::reussir::ReussirDialect"];
}

def ReussirInferUnionTag : Pass<"reussir-infer-union-tag", "::mlir::func::FuncOp"> {
  let summary = "Infer union tags for Reussir operations";
  let description = [{
    This pass is used to infer union tags for Reussir operations.
  }];
  let constructor = "::mlir::reussir::createReussirInferUnionTagPass()";
  let dependentDialects = ["::mlir::reussir::ReussirDialect"];
}

def ReussirPrintReuseAnalysis : Pass<"reussir-print-reuse-analysis", "::mlir::ModuleOp"> {
  let summary = "Print Reuse Analysis";
  let description = [{
    This pass is used to print the reuse analysis information.
  }];
  let constructor = "::mlir::reussir::createReussirPrintReuseAnalysisPass()";
  let dependentDialects = ["::mlir::reussir::ReussirDialect"];
}

def ReussirTokenReuse : Pass<"reussir-token-reuse", "::mlir::ModuleOp"> {
  let summary = "Apply reuse analysis to reuse token and recyle unused tokens";
  let description = [{
    This pass is used to apply reuse analysis to reuse token and recyle unused tokens.
  }];
  let constructor = "::mlir::reussir::createReussirTokenReusePass()";
  let dependentDialects = ["::mlir::reussir::ReussirDialect"];
}

#endif // REUSSIR_PASSES
