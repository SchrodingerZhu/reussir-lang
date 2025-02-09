#ifndef REUSSIR_COMPOSITE_LAYOUT_INTERFACE
#define REUSSIR_COMPOSITE_LAYOUT_INTERFACE

include "mlir/IR/OpBase.td"

def ReussirCompositeLayoutInterface : TypeInterface<"ReussirCompositeLayoutInterface"> {
  let description = [{
    Unified interface for converting type to LLVM struct layout.
  }];
  let cppNamespace = "::mlir::reussir";
  let methods = [
    InterfaceMethod<[{
        Get the CompositeLayout of the type.
      }],
      /*retTy=*/"::mlir::reussir::CompositeLayout",
      /*methodName=*/"getCompositeLayout",
      /*args=*/(ins "::mlir::DataLayout":$dataLayout)
    >
  ];
}
#endif // REUSSIR_COMPOSITE_LAYOUT_INTERFACE
