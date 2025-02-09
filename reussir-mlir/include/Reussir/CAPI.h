#include <mlir-c/AffineExpr.h>
#include <mlir-c/AffineMap.h>
#include <mlir-c/BuiltinAttributes.h>
#include <mlir-c/BuiltinTypes.h>
#include <mlir-c/Conversion.h>
#include <mlir-c/Debug.h>
#include <mlir-c/Diagnostics.h>
#include <mlir-c/Dialect/Arith.h>
#include <mlir-c/Dialect/Async.h>
#include <mlir-c/Dialect/ControlFlow.h>
#include <mlir-c/Dialect/Func.h>
#include <mlir-c/Dialect/GPU.h>
#include <mlir-c/Dialect/LLVM.h>
#include <mlir-c/Dialect/Linalg.h>
#include <mlir-c/Dialect/MLProgram.h>
#include <mlir-c/Dialect/Math.h>
#include <mlir-c/Dialect/MemRef.h>
#include <mlir-c/Dialect/NVGPU.h>
#include <mlir-c/Dialect/OpenMP.h>
#include <mlir-c/Dialect/PDL.h>
#include <mlir-c/Dialect/Quant.h>
#include <mlir-c/Dialect/ROCDL.h>
#include <mlir-c/Dialect/SCF.h>
#include <mlir-c/Dialect/SPIRV.h>
#include <mlir-c/Dialect/Shape.h>
#include <mlir-c/Dialect/SparseTensor.h>
#include <mlir-c/Dialect/Tensor.h>
#include <mlir-c/Dialect/Transform.h>
#include <mlir-c/Dialect/Vector.h>
#include <mlir-c/ExecutionEngine.h>
#include <mlir-c/IR.h>
#include <mlir-c/IntegerSet.h>
#include <mlir-c/Interfaces.h>
#include <mlir-c/Pass.h>
#include <mlir-c/RegisterEverything.h>
#include <mlir-c/Support.h>
#include <mlir-c/Target/LLVMIR.h>
#include <mlir-c/Transforms.h>

#ifdef __cplusplus
extern "C" {
#endif
MLIR_DECLARE_CAPI_DIALECT_REGISTRATION(Reussir, reussir);

enum Linkage : unsigned {
  PRIVATE,
  INTERNAL,
  AVAILABLE_EXTERNALLY,
  LINKONCE,
  WEAK,
  COMMON,
  APPENDING,
  EXTERN_WEAK,
  LINKONCE_ODR,
  WEAK_ODR,
  EXTERNAL
};

void reuseIRSetLinkageForFunc(MlirOperation, enum Linkage);

MlirAttribute reuseIRFreezingKindGetNonfreezing(MlirContext);
MlirAttribute reuseIRFreezingKindGetUnfrozen(MlirContext);
MlirAttribute reuseIRFreezingKindGetFrozen(MlirContext);

MlirAttribute reuseIRAtomicKindGetNonatomic(MlirContext);
MlirAttribute reuseIRAtomicKindGetAtomic(MlirContext);

MlirType reuseIRGetRcType(MlirType inner, MlirAttribute atomicKind,
                          MlirAttribute freezeKind);
MlirType reuseIRGetRefType(MlirType inner, MlirAttribute freezeKind);
MlirType reuseIRGetMRefType(MlirType inner, MlirAttribute atomicKind);
MlirType reuseIRGetNullableType(MlirType inner);
MlirType reuseIRGetCompositeType(MlirContext context, MlirStringRef name);
void reuseIRCompleteCompositeType(MlirType compositeType, size_t numInnerTypes,
                                  const MlirType *innerTypes);
MlirType reuseIRGetUnionType(MlirContext context, MlirStringRef name);
void reuseIRCompleteUnionType(MlirType unionType, size_t numInnerTypes,
                              const MlirType *innerTypes);
#ifdef __cplusplus
}
#endif
