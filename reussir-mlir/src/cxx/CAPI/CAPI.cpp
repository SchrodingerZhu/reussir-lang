#include "Reussir/CAPI.h"
#include "Reussir/IR/ReussirDialect.h"
#include "Reussir/IR/ReussirOps.h"
#include "Reussir/IR/ReussirOpsEnums.h"
#include "mlir/CAPI/Pass.h"
#include "mlir/CAPI/Registration.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMAttrs.h"
#include "mlir/IR/Attributes.h"
namespace mlir {
namespace reussir {
extern "C" {
MLIR_DEFINE_CAPI_DIALECT_REGISTRATION(Reussir, reussir, ReussirDialect)

// polyfill for pre-20 LLVM
void reuseIRSetLinkageForFunc(MlirOperation op, Linkage linkage) {
  Operation *funcOp = cast<func::FuncOp>(unwrap(op));
  mlir::LLVM::Linkage llvmLinkage;
  switch (linkage) {
  case PRIVATE:
    llvmLinkage = mlir::LLVM::Linkage::Private;
    break;
  case INTERNAL:
    llvmLinkage = mlir::LLVM::Linkage::Internal;
    break;
  case AVAILABLE_EXTERNALLY:
    llvmLinkage = mlir::LLVM::Linkage::AvailableExternally;
    break;
  case LINKONCE:
    llvmLinkage = mlir::LLVM::Linkage::Linkonce;
    break;
  case WEAK:
    llvmLinkage = mlir::LLVM::Linkage::Weak;
    break;
  case COMMON:
    llvmLinkage = mlir::LLVM::Linkage::Common;
    break;
  case APPENDING:
    llvmLinkage = mlir::LLVM::Linkage::Appending;
    break;
  case EXTERN_WEAK:
    llvmLinkage = mlir::LLVM::Linkage::ExternWeak;
    break;
  case LINKONCE_ODR:
    llvmLinkage = mlir::LLVM::Linkage::LinkonceODR;
    break;
  case WEAK_ODR:
    llvmLinkage = mlir::LLVM::Linkage::WeakODR;
    break;
  case EXTERNAL:
    llvmLinkage = mlir::LLVM::Linkage::External;
    break;
  }
  funcOp->setAttr("llvm.linkage", mlir::LLVM::LinkageAttr::get(
                                      funcOp->getContext(), llvmLinkage));
}

MlirAttribute reuseIRFreezingKindGetNonfreezing(MlirContext context) {
  return wrap(
      FreezingKindAttr::get(unwrap(context), FreezingKind::nonfreezing));
}
MlirAttribute reuseIRFreezingKindGetUnfrozen(MlirContext context) {
  return wrap(FreezingKindAttr::get(unwrap(context), FreezingKind::unfrozen));
}
MlirAttribute reuseIRFreezingKindGetFrozen(MlirContext context) {
  return wrap(FreezingKindAttr::get(unwrap(context), FreezingKind::frozen));
}

MlirAttribute reuseIRAtomicKindGetNonatomic(MlirContext context) {
  return wrap(AtomicKindAttr::get(unwrap(context), AtomicKind::nonatomic));
}
MlirAttribute reuseIRAtomicKindGetAtomic(MlirContext context) {
  return wrap(AtomicKindAttr::get(unwrap(context), AtomicKind::atomic));
}
MlirType reuseIRGetRcType(MlirType inner, MlirAttribute atomicKind,
                          MlirAttribute freezeKind) {
  auto innerTy = unwrap(inner);
  return wrap(RcType::get(innerTy.getContext(), unwrap(inner),
                          cast<AtomicKindAttr>(unwrap(atomicKind)),
                          cast<FreezingKindAttr>(unwrap(freezeKind))));
}
MlirType reuseIRGetRefType(MlirType inner, MlirAttribute freezeKind) {
  auto innerTy = unwrap(inner);
  return wrap(RefType::get(innerTy.getContext(), unwrap(inner),
                           cast<FreezingKindAttr>(unwrap(freezeKind))));
}
MlirType reuseIRGetMRefType(MlirType inner, MlirAttribute atomicKind) {
  auto innerTy = unwrap(inner);
  return wrap(MRefType::get(innerTy.getContext(), unwrap(inner),
                            cast<AtomicKindAttr>(unwrap(atomicKind))));
}
MlirType reuseIRGetNullableType(MlirType inner) {
  auto innerTy = unwrap(inner);
  return wrap(NullableType::get(innerTy.getContext(), unwrap(inner)));
}
MlirType reuseIRGetCompositeType(MlirContext context, MlirStringRef name) {
  auto ref = unwrap(name);
  auto attr = StringAttr::get(unwrap(context), ref);
  return wrap(CompositeType::get(unwrap(context), attr));
}
void reuseIRCompleteCompositeType(MlirType compositeType, size_t numInnerTypes,
                                  const MlirType *innerTypes) {
  auto compositeTy = cast<CompositeType>(unwrap(compositeType));
  llvm::SmallVector<Type> innerTys;
  auto ref = unwrapList(numInnerTypes, innerTypes, innerTys);
  compositeTy.complete(ref);
}
MlirType reuseIRGetUnionType(MlirContext context, MlirStringRef name) {
  auto ref = unwrap(name);
  auto attr = StringAttr::get(unwrap(context), ref);
  return wrap(UnionType::get(unwrap(context), attr));
}
void reuseIRCompleteUnionType(MlirType unionType, size_t numInnerTypes,
                              const MlirType *innerTypes) {
  auto unionTy = cast<UnionType>(unwrap(unionType));
  llvm::SmallVector<Type> innerTys;
  auto ref = unwrapList(numInnerTypes, innerTypes, innerTys);
  unionTy.complete(ref);
}
} // extern "C"
} // namespace reussir
} // namespace mlir
