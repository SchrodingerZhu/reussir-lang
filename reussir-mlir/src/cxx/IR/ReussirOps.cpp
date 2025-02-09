#include "Reussir/IR/ReussirOps.h"
#include "Reussir/Common.h"
#include "Reussir/IR/ReussirOpsEnums.h"
#include "Reussir/IR/ReussirTypes.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/IR/TypeRange.h"
#include "mlir/IR/Value.h"
#include "mlir/Support/LogicalResult.h"
#include "llvm/Support/ErrorHandling.h"
#include <algorithm>

namespace mlir {
namespace REUSSIR_DECL_SCOPE {
// BorrowOp
mlir::reussir::LogicalResult RcBorrowOp::verify() {
  RcType input = getObject().getType();
  RefType output = getType();
  if (input.getFreezingKind() != output.getFreezingKind())
    return emitError("the borrowed reference must have the consistent "
                     "freezing state with the RC pointer");
  if (input.getPointee() != output.getPointee())
    return emitError("the borrowed reference must have the consistent "
                     "pointee type with the RC pointer");
  return mlir::reussir::success();
}
// RcAcquireOp
mlir::reussir::LogicalResult RcAcquireOp::verify() {
  RcType rcPtrTy = getRcPtr().getType();
  if (rcPtrTy.getFreezingKind().getValue() == FreezingKind::unfrozen)
    return emitOpError("cannot be applied to an unfrozen RC pointer");
  return mlir::reussir::success();
}
// ValueToRefOp
mlir::reussir::LogicalResult ValueToRefOp::verify() {
  if (getResult().getType().getFreezingKind().getValue() !=
      FreezingKind::nonfreezing)
    return emitOpError("must return a nonfreezing reference");
  if (getValue().getType() != getResult().getType().getPointee())
    return emitOpError("must return a reference whose pointee is of the "
                       "same type of the input");
  return mlir::reussir::success();
}

// ProjOp
mlir::reussir::LogicalResult ProjOp::verify() {
  RefType input = getObject().getType();
  if (input.getFreezingKind() != getType().getFreezingKind())
    return emitOpError(
        "must return a reference with the same freezing kind as the input");
  if (!isProjectable(input.getPointee()))
    return emitOpError(
        "must operate on a reference to a composite or an array");
  auto targetType =
      llvm::TypeSwitch<mlir::Type, mlir::Type>(input.getPointee())
          .Case<CompositeType>([&](const CompositeType &ty) {
            size_t size = ty.getInnerTypes().size();
            return getIndex().getZExtValue() < size
                       ? ty.getInnerTypes()[getIndex().getZExtValue()]
                       : mlir::Type{};
          })
          .Case<ArrayType>([&](const ArrayType &ty) {
            if (getIndex().getZExtValue() >= ty.getSizes()[0])
              return mlir::Type{};
            if (ty.getSizes().size() == 1)
              return ty.getElementType();
            return cast<mlir::Type>(ArrayType::get(
                getContext(), ty.getElementType(), ty.getSizes().drop_front()));
          })
          .Default([](auto &&) { return mlir::Type{}; });
  if (!targetType)
    return emitOpError("cannot project with an out-of-bound index");
  if (targetType != getType().getPointee())
    return emitOpError("expected to return a reference to ")
           << targetType << ", but found a reference to "
           << getType().getPointee();

  return mlir::reussir::success();
}

// LoadOp
mlir::reussir::LogicalResult LoadOp::verify() {
  RefType input = getObject().getType();
  mlir::Type targetType{};
  if (auto mref = dyn_cast<MRefType>(input.getPointee())) {
    if (input.getFreezingKind().getValue() == FreezingKind::nonfreezing)
      return emitOpError(
          "cannot load a mutable RC pointer through a nonfreezing reference");
    targetType = NullableType::get(getContext(),
                                   RcType::get(getContext(), mref.getPointee(),
                                               mref.getAtomicKind(),
                                               input.getFreezingKind()));
  } else
    targetType = input.getPointee();
  if (targetType != getType())
    return emitOpError("expected to return a value of ")
           << targetType << ", but " << getType() << " is found instead";
  return mlir::reussir::success();
}

// ClosureNewOp
ClosureType ClosureNewOp::getClosureType() {
  return llvm::TypeSwitch<mlir::Type, ClosureType>(getClosure().getType())
      .Case<RcType>(
          [](const RcType &ty) { return cast<ClosureType>(ty.getPointee()); })
      .Default([](const mlir::Type &ty) { return cast<ClosureType>(ty); });
}

mlir::reussir::LogicalResult ClosureNewOp::verify() {
  if (getNumRegions() > 1)
    return emitOpError("cannot have more than one region");
  Region *region = &getRegion(0);
  ClosureType closureTy = getClosureType();
  if (region->getArguments().size() != closureTy.getInputTypes().size())
    return emitOpError("the number of arguments in the region must match the "
                       "number of input types in the closure type");
  if (std::any_of(region->getArguments().begin(), region->getArguments().end(),
                  [&](BlockArgument arg) {
                    return arg.getType() !=
                           closureTy.getInputTypes()[arg.getArgNumber()];
                  }))
    return emitOpError("the types of arguments in the region must match the "
                       "input types in the closure type");
  return mlir::reussir::success();
}

// ClosureYieldOp
mlir::reussir::LogicalResult ClosureYieldOp::verify() {
  ClosureNewOp op = getParentOp();
  ClosureType closureTy = op.getClosureType();
  if (getValue() && !closureTy.getOutputType())
    return emitOpError("cannot yield a value in a closure without output");
  if (!getValue() && closureTy.getOutputType())
    return emitOpError("must yield a value in a closure with output");
  if (getValue().getType() != closureTy.getOutputType())
    return emitOpError("expected to yield a value of ")
           << closureTy.getOutputType() << ", but " << getValue().getType()
           << " is found instead";
  return mlir::reussir::success();
}

template <typename EmitError>
static LogicalResult verifyTokenForRC(Operation *op, TokenType token, RcType rc,
                                      EmitError emitOpError,
                                      bool isReturn = false) {
  auto module = op->getParentOfType<ModuleOp>();
  if (!module)
    return emitOpError("cannot find the module containing the operation");
  DataLayout dataLayout{module};
  auto rcBoxTy = RcBoxType::get(op->getContext(), rc.getPointee(),
                                rc.getAtomicKind(), rc.getFreezingKind());
  auto size = dataLayout.getTypeSize(rcBoxTy);
  auto align = dataLayout.getTypeABIAlignment(rcBoxTy);
  if (token.getAlignment() != align)
    return emitOpError("expected")
           << (isReturn ? " to return " : " ")
           << "a nullable token of alignment " << align
           << ", but found a nullable token of alignment "
           << token.getAlignment();
  if (token.getSize() != size)
    return emitOpError("expected")
           << (isReturn ? " to return " : " ") << "a nullable token of size "
           << size.getFixedValue() << ", but found a nullable token of size "
           << token.getSize();
  return success();
}

// RcReleaseOp
mlir::reussir::LogicalResult RcReleaseOp::verify() {
  RcType rcPtrTy = getRcPtr().getType();
  if (rcPtrTy.getFreezingKind().getValue() == FreezingKind::unfrozen)
    return emitOpError("cannot be applied to an unfrozen RC pointer");
  if (rcPtrTy.getFreezingKind().getValue() == FreezingKind::frozen &&
      getNumResults() > 0)
    return emitOpError("cannot have any result when applied to a frozen RC "
                       "pointer");
  if (rcPtrTy.getFreezingKind().getValue() == FreezingKind::nonfreezing &&
      getNumResults() != 1)
    return emitOpError("must have a result when applied to a nonfreezing RC "
                       "pointer");
  // get current module
  if (!getToken())
    return mlir::reussir::success();
  auto tokenTy = cast<TokenType>(getToken().getType().getPointer());
  return verifyTokenForRC(
      this->getOperation(), tokenTy, rcPtrTy,
      [&](const Twine &msg) { return emitOpError(msg); }, true);
}

// RcDecreaseOp
mlir::reussir::LogicalResult RcDecreaseOp::verify() {
  RcType rcPtrTy = getRcPtr().getType();
  if (rcPtrTy.getFreezingKind().getValue() != FreezingKind::nonfreezing)
    return emitOpError("can only be applied to a nonfreezing RC pointer");
  return mlir::reussir::success();
}

// CompositeAssembleOp
mlir::reussir::LogicalResult CompositeAssembleOp::verify() {
  CompositeType compositeTy = dyn_cast<CompositeType>(getType());
  if (!compositeTy)
    return emitOpError("must return a composite type");
  if (!compositeTy.isComplete())
    return emitOpError("cannot assemble an incomplete composite type");
  if (compositeTy.getInnerTypes().size() != getNumOperands())
    return emitOpError("the number of operands must match the number of "
                       "fields in the composite type");
  for (size_t i = 0; i < getNumOperands(); ++i) {
    if (auto mrefTy = dyn_cast<MRefType>(compositeTy.getInnerTypes()[i])) {
      auto rcTy = RcType::get(
          getContext(), mrefTy.getPointee(), mrefTy.getAtomicKind(),
          FreezingKindAttr::get(getContext(), FreezingKind::unfrozen));
      auto nullableTy = NullableType::get(getContext(), rcTy);
      if (getOperand(i).getType() != nullableTy)
        return emitOpError("the type of operand #")
               << i << " must be " << nullableTy;
      continue;
    }
    if (getOperand(i).getType() != compositeTy.getInnerTypes()[i])
      return emitOpError("the type of operand #")
             << i << " must match the type of the field in the composite type";
  }
  return mlir::reussir::success();
}

// UnionAssembleOp
mlir::reussir::LogicalResult UnionAssembleOp::verify() {
  UnionType unionTy = dyn_cast<UnionType>(getType());
  if (!unionTy)
    return emitOpError("must return a union type");
  if (!unionTy.isComplete())
    return emitOpError("cannot assemble an incomplete union type");
  if (getTag().getZExtValue() >= unionTy.getInnerTypes().size())
    return emitOpError("the tag must be within the range of the union type");
  if (getOperand().getType() !=
      unionTy.getInnerTypes()[getTag().getZExtValue()])
    return emitOpError("the type of the operand must match the type of the "
                       "field in the union type corresponding to the tag");
  return mlir::reussir::success();
}

// RcCreateOp
mlir::reussir::LogicalResult RcCreateOp::verify() {
  RcType rcPtrTy = getType();
  switch (rcPtrTy.getFreezingKind().getValue()) {
  case FreezingKind::frozen:
    return emitOpError("cannot create a frozen RC pointer");
  case FreezingKind::nonfreezing:
    if (getRegion())
      return emitOpError("cannot have a region when creating a nonfreezing RC "
                         "pointer");
    break;
  case FreezingKind::unfrozen:
    if (!getRegion())
      return emitOpError("must have a region when creating an unfrozen RC "
                         "pointer");
    break;
  }
  auto tokenTy = cast<TokenType>(getToken().getType());
  return verifyTokenForRC(this->getOperation(), tokenTy, rcPtrTy,
                          [&](const Twine &msg) { return emitOpError(msg); });

  return mlir::reussir::success();
}
template <StringLiteral Literal> struct ParseKeywordAsUnitAttr {
  OptionalParseResult operator()(OpAsmParser &parser, UnitAttr &attr) {
    if (parser.parseOptionalKeyword(Literal).succeeded())
      attr = UnitAttr::get(parser.getContext());
    return success();
  }
};

template <StringLiteral Literal> struct PrintKeywordAsUnitAttr {
  void operator()(OpAsmPrinter &printer, Operation *, const UnitAttr &attr) {
    if (attr)
      printer.printKeywordOrString(Literal);
  }
};

// RegionRunOp (RegionBranchOpInterface)
void RegionRunOp::getSuccessorRegions(
    RegionBranchPoint point, SmallVectorImpl<RegionSuccessor> &regions) {
  // If the predecessor is the ExecuteRegionOp, branch into the body.
  if (point.isParent()) {
    regions.push_back(RegionSuccessor(&getBody()));
    return;
  }
  // Otherwise, the region branches back to the parent operation.
  regions.push_back(RegionSuccessor(getResults()));
}
// RegionRunOp verification
mlir::LogicalResult RegionRunOp::verify() {
  // The region cannot be nested inside a region directly.
  // The region must contains MLIR region whose argument is a RegionCtxType.
  if (this->getOperation()->getParentOfType<RegionRunOp>())
    return emitOpError("cannot be directly nested inside a region");
  if (this->getBody().getArguments().size() != 1)
    return emitOpError("must have exactly one argument");
  if (!isa<RegionCtxType>(this->getBody().getArgument(0).getType()))
    return emitOpError("the argument must be of type RegionCtxType");
  return mlir::success();
}

mlir::LogicalResult RegionYieldOp::verify() {
  // If the parent operation has a return type, the operation must yield a value
  // of the same type. Otherwise, the operation must not yield a value.
  RegionRunOp parentOp = getParentOp();
  if (auto result = parentOp.getResult()) {
    if (!getValue())
      return emitOpError("must yield a value");
    if (getValue().getType() != result.getType())
      return emitOpError("expected to yield a value of ")
             << result.getType() << ", but found a value of "
             << getValue().getType();
    if (auto rcTy = dyn_cast<RcType>(getValue().getType())) {
      if (rcTy.getFreezingKind().getValue() == FreezingKind::unfrozen)
        return emitOpError(
            "cannot have an unfrozen RC pointer escaped from the region");
    }
  } else if (getValue())
    return emitOpError("cannot yield a value when the parent operation does "
                       "not have a return type");
  return mlir::success();
}

// MRefAssignOp verification
mlir::LogicalResult MRefAssignOp::verify() {
  auto rcTy = dyn_cast<RcType>(getValue().getType().getPointer());
  if (!rcTy || rcTy.getFreezingKind().getValue() != FreezingKind::unfrozen)
    return emitOpError(
        "the value for assignment must be an unfrozen RC pointer");
  auto mrefTy = dyn_cast<MRefType>(getRefOfMRef().getType().getPointee());
  if (!mrefTy)
    return emitOpError("the reference must be pointing to a MRefType");
  if (mrefTy.getPointee() != rcTy.getPointee())
    return emitOpError("the pointee type of the reference must match the "
                       "pointee type of the value");
  return mlir::success();
}

// RcFreezeOp verification
mlir::LogicalResult RcFreezeOp::verify() {
  RcType rcPtrTy = getRcPtr().getType();
  if (rcPtrTy.getFreezingKind().getValue() != FreezingKind::unfrozen)
    return emitOpError("can only be applied to an unfrozen RC pointer");
  if (rcPtrTy.getPointee() != getType().getPointee() ||
      rcPtrTy.getAtomicKind() != getType().getAtomicKind())
    return emitOpError("must return a RC pointer with the same pointee type "
                       "and atomic kind as the input");
  if (getType().getFreezingKind().getValue() != FreezingKind::frozen)
    return emitOpError("must return a frozen RC pointer");
  return mlir::success();
}

// ClosureApplyOp verification
mlir::LogicalResult ClosureApplyOp::verify() {
  RcType inputTy = getClosure().getType();
  ClosureType inputClosure = dyn_cast<ClosureType>(inputTy.getPointee());
  RcType outputTy = getType();
  ClosureType outputClosure = dyn_cast<ClosureType>(outputTy.getPointee());
  if (!inputClosure)
    return emitOpError("must operate on a RC pointer to a closure type");
  if (!outputClosure)
    return emitOpError("must return a RC pointer to a closure type");
  TypeRange allInputTypes = inputClosure.getInputTypes();
  TypeRange actualInputTypes = getArgs().getTypes();
  if (actualInputTypes.size() > allInputTypes.size())
    return emitOpError("too many arguments");
  TypeRange slicedInputTypes =
      allInputTypes.take_front(actualInputTypes.size());
  if (actualInputTypes != slicedInputTypes)
    return emitOpError(
        "the types of arguments must match the leading input types in "
        "the closure type");
  TypeRange afterInputTypes = allInputTypes.drop_front(actualInputTypes.size());
  SmallVector<Type> afterInputTypesVec(afterInputTypes.begin(),
                                       afterInputTypes.end());
  ClosureType expectedOutputClosure = ClosureType::get(
      getContext(), afterInputTypesVec, inputClosure.getOutputType());
  if (outputClosure != expectedOutputClosure)
    return emitOpError("expected to return a RC pointer to ")
           << expectedOutputClosure << ", but found a RC pointer to "
           << outputClosure;
  return mlir::success();
}
// ClosureEvalOp verification
mlir::LogicalResult ClosureEvalOp::verify() {
  RcType inputTy = getClosure().getType();
  ClosureType inner = dyn_cast<ClosureType>(inputTy.getPointee());
  if (!inner)
    return emitOpError("must evaluate a RC pointer to a closure type");
  if (inner.getInputTypes().size() != 0)
    return emitOpError("must evaluate a fully applied closure");
  if (inner.getOutputType() != getType())
    return emitOpError("expected to return a value of ")
           << inner.getOutputType() << ", but " << getType()
           << " is found instead";
  return mlir::success();
}
// RcIsUniqueOp verification
mlir::LogicalResult RcIsUniqueOp::verify() {
  if (getRc().getType().getFreezingKind().getValue() !=
      FreezingKind::nonfreezing)
    return emitOpError("can only be applied to a nonfreezing RC pointer");
  return mlir::success();
}
// RcUniquifyOp verification
mlir::LogicalResult RcUniquifyOp::verify() {
  if (getRc().getType().getFreezingKind().getValue() !=
      FreezingKind::nonfreezing)
    return emitOpError("can only be applied to a nonfreezing RC pointer");
  return mlir::success();
}
} // namespace REUSSIR_DECL_SCOPE
} // namespace mlir

#define GET_OP_CLASSES
#include "Reussir/IR/ReussirOps.cpp.inc"
