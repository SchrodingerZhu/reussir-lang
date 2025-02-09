#ifndef REUSSIR_MANGLE_INTERFACE
#define REUSSIR_MANGLE_INTERFACE

include "mlir/IR/OpBase.td"

def ReussirMangleInterface : TypeInterface<"ReussirMangleInterface"> {
  let description = [{
    Mangle the type so that we have a unique name for its associated functions.
  }];
  let cppNamespace = "::mlir::reussir";
  let methods = [
    InterfaceMethod<[{
        Return the mangled name of the type.
      }],
      /*retTy=*/"void",
      /*methodName=*/"formatMangledNameTo",
      /*args=*/(ins "::llvm::raw_ostream &":$buffer)
    >
  ];
}
#endif // REUSSIR_MANGLE_INTERFACE
