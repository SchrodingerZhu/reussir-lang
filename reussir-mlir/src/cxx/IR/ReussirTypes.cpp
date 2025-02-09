#include "Reussir/IR/ReussirTypes.h"
#include "Reussir/Common.h"
#include "Reussir/IR/ReussirDialect.h"

#include "Reussir/IR/ReussirOpsEnums.h"
#include "Reussir/Interfaces/ReussirCompositeLayoutInterface.h"
#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMTypes.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/Interfaces/DataLayoutInterfaces.h"
#include "mlir/Support/LLVM.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/IR/DataLayout.h"
#include "llvm/Support/Alignment.h"
#include "llvm/Support/MathExtras.h"
#include "llvm/Support/TypeSize.h"
#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <iterator>
#include <numeric>
#include <string>

namespace mlir {
namespace REUSSIR_DECL_SCOPE {
static LogicalResult parseAtomicKind(AsmParser &parser, AtomicKindAttr &attr) {
  std::string buffer;
  if (parser.parseKeywordOrString(&buffer).succeeded()) {
    if (auto kind = symbolizeAtomicKind(buffer)) {
      attr = AtomicKindAttr::get(parser.getContext(), *kind);
      return LogicalResult::success();
    }
  }
  return parser.emitError(parser.getCurrentLocation(),
                          "failed to parse atomic kind");
}
static LogicalResult parseFreezingKind(AsmParser &parser,
                                       FreezingKindAttr &attr) {
  std::string buffer;
  if (parser.parseKeywordOrString(&buffer).succeeded()) {
    if (auto kind = symbolizeFreezingKind(buffer)) {
      attr = FreezingKindAttr::get(parser.getContext(), *kind);
      return LogicalResult::success();
    }
  }
  return parser.emitError(parser.getCurrentLocation(),
                          "failed to parse freezing kind");
}

static void printAtomicKind(AsmPrinter &printer, const AtomicKindAttr &attr) {
  printer.printKeywordOrString(stringifyAtomicKind(attr.getValue()));
}

static void printFreezingKind(AsmPrinter &printer,
                              const FreezingKindAttr &attr) {
  printer.printKeywordOrString(stringifyFreezingKind(attr.getValue()));
}
} // namespace REUSSIR_DECL_SCOPE
} // namespace mlir

#define GET_TYPEDEF_CLASSES
#include "Reussir/IR/ReussirOpsTypes.cpp.inc"

namespace mlir {
namespace REUSSIR_DECL_SCOPE {
#pragma push_macro("GENERATE_POINTER_ALIKE_LAYOUT")
#define GENERATE_POINTER_ALIKE_LAYOUT(TYPE)                                    \
  ::llvm::TypeSize TYPE::getTypeSizeInBits(                                    \
      const ::mlir::DataLayout &dataLayout,                                    \
      [[maybe_unused]] ::mlir::DataLayoutEntryListRef params) const {          \
    return dataLayout.getTypeSizeInBits(                                       \
        mlir::LLVM::LLVMPointerType::get(getContext()));                       \
  };                                                                           \
  uint64_t TYPE::getABIAlignment(                                              \
      const ::mlir::DataLayout &dataLayout,                                    \
      [[maybe_unused]] ::mlir::DataLayoutEntryListRef params) const {          \
    return dataLayout.getTypeABIAlignment(                                     \
        mlir::LLVM::LLVMPointerType::get(getContext()));                       \
  }                                                                            \
  uint64_t TYPE::getPreferredAlignment(const ::mlir::DataLayout &dataLayout,   \
                                       ::mlir::DataLayoutEntryListRef params)  \
      const {                                                                  \
    return dataLayout.getTypePreferredAlignment(                               \
        mlir::LLVM::LLVMPointerType::get(getContext()));                       \
  }
GENERATE_POINTER_ALIKE_LAYOUT(RcType)
GENERATE_POINTER_ALIKE_LAYOUT(RefType)
GENERATE_POINTER_ALIKE_LAYOUT(TokenType)
GENERATE_POINTER_ALIKE_LAYOUT(MRefType)
GENERATE_POINTER_ALIKE_LAYOUT(RegionCtxType)
GENERATE_POINTER_ALIKE_LAYOUT(NullableType)
GENERATE_POINTER_ALIKE_LAYOUT(PtrType)
#pragma pop_macro("GENERATE_POINTER_ALIKE_LAYOUT")

// RcBox DataLayoutInterface:
::llvm::TypeSize RcBoxType::getTypeSizeInBits(
    const ::mlir::DataLayout &dataLayout,
    [[maybe_unused]] ::mlir::DataLayoutEntryListRef params) const {
  return getCompositeLayout(dataLayout).getSize() * 8;
};

uint64_t RcBoxType::getABIAlignment(
    const ::mlir::DataLayout &dataLayout,
    [[maybe_unused]] ::mlir::DataLayoutEntryListRef params) const {
  return getCompositeLayout(dataLayout).getAlignment().value();
}
uint64_t
RcBoxType::getPreferredAlignment(const ::mlir::DataLayout &dataLayout,
                                 ::mlir::DataLayoutEntryListRef params) const {
  return getCompositeLayout(dataLayout).getAlignment().value();
}

// Array DataLayoutInterface:
::llvm::TypeSize ArrayType::getTypeSizeInBits(
    const ::mlir::DataLayout &dataLayout,
    [[maybe_unused]] ::mlir::DataLayoutEntryListRef params) const {
  uint64_t numOfElems = std::reduce(getSizes().begin(), getSizes().end(), 1,
                                    std::multiplies<uint64_t>());
  return dataLayout.getTypeSizeInBits(getElementType()) * numOfElems;
}

uint64_t ArrayType::getABIAlignment(
    const ::mlir::DataLayout &dataLayout,
    [[maybe_unused]] ::mlir::DataLayoutEntryListRef params) const {
  return dataLayout.getTypeABIAlignment(getElementType());
}

uint64_t
ArrayType::getPreferredAlignment(const ::mlir::DataLayout &dataLayout,
                                 ::mlir::DataLayoutEntryListRef params) const {
  return dataLayout.getTypePreferredAlignment(getElementType());
}

// Vector DataLayoutInterface:
::llvm::TypeSize VectorType::getTypeSizeInBits(
    const ::mlir::DataLayout &dataLayout,
    [[maybe_unused]] ::mlir::DataLayoutEntryListRef params) const {
  return getCompositeLayout(dataLayout).getSize() * 8;
}

uint64_t VectorType::getABIAlignment(
    const ::mlir::DataLayout &dataLayout,
    [[maybe_unused]] ::mlir::DataLayoutEntryListRef params) const {
  return getCompositeLayout(dataLayout).getAlignment().value();
}

uint64_t
VectorType::getPreferredAlignment(const ::mlir::DataLayout &dataLayout,
                                  ::mlir::DataLayoutEntryListRef params) const {
  return getCompositeLayout(dataLayout).getAlignment().value();
}

// Opaque DataLayoutInterface:
::llvm::TypeSize OpaqueType::getTypeSizeInBits(
    const ::mlir::DataLayout &dataLayout,
    [[maybe_unused]] ::mlir::DataLayoutEntryListRef params) const {
  return getCompositeLayout(dataLayout).getSize() * 8;
}

uint64_t OpaqueType::getABIAlignment(
    const ::mlir::DataLayout &dataLayout,
    [[maybe_unused]] ::mlir::DataLayoutEntryListRef params) const {
  return getCompositeLayout(dataLayout).getAlignment().value();
}

uint64_t
OpaqueType::getPreferredAlignment(const ::mlir::DataLayout &dataLayout,
                                  ::mlir::DataLayoutEntryListRef params) const {
  return getCompositeLayout(dataLayout).getAlignment().value();
}

// Closure DataLayoutInterface:
::llvm::TypeSize ClosureType::getTypeSizeInBits(
    const ::mlir::DataLayout &dataLayout,
    [[maybe_unused]] ::mlir::DataLayoutEntryListRef params) const {
  return getCompositeLayout(dataLayout).getSize() * 8;
}

uint64_t ClosureType::getABIAlignment(
    const ::mlir::DataLayout &dataLayout,
    [[maybe_unused]] ::mlir::DataLayoutEntryListRef params) const {
  return getCompositeLayout(dataLayout).getAlignment().value();
}

uint64_t ClosureType::getPreferredAlignment(
    const ::mlir::DataLayout &dataLayout,
    ::mlir::DataLayoutEntryListRef params) const {
  return getCompositeLayout(dataLayout).getAlignment().value();
}

// Token Verifier:
::mlir::reussir::LogicalResult
TokenType::verify(::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError,
                  size_t alignment, size_t size) {
  if (!llvm::isPowerOf2_64(alignment))
    return emitError() << "alignment must be a power of 2";
  if (size == 0)
    return emitError() << "size must be non-zero";
  if (size % alignment != 0)
    return emitError() << "size must be a multiple of alignment";
  return ::mlir::reussir::success();
}

// TokenType mangle
void TokenType::formatMangledNameTo(::llvm::raw_ostream &buffer) const {
  buffer << "$Token<" << getSize() << "," << getAlignment() << ">";
}

// RcType mangle
void RcType::formatMangledNameTo(::llvm::raw_ostream &buffer) const {
  buffer << "$Rc<";
  mlir::reussir::formatMangledNameTo(getPointee(), buffer);
  buffer << "," << stringifyAtomicKind(getAtomicKind().getValue()) << ","
         << stringifyFreezingKind(getFreezingKind().getValue()) << ">";
}

// ArrayType mangle
void ArrayType::formatMangledNameTo(::llvm::raw_ostream &buffer) const {
  buffer << "$Array<";
  mlir::reussir::formatMangledNameTo(getElementType(), buffer);
  buffer << ",";
  llvm::interleaveComma(getSizes(), buffer);
  buffer << ">";
}

// VectorType mangle
void VectorType::formatMangledNameTo(::llvm::raw_ostream &buffer) const {
  buffer << "$Vector<";
  mlir::reussir::formatMangledNameTo(getElementType(), buffer);
  buffer << ">";
}

// OpaqueType mangle
void OpaqueType::formatMangledNameTo(::llvm::raw_ostream &buffer) const {
  buffer << "$Opaque<" << getSize() << "," << getAlignment() << ">";
}

// ClosureType mangle
void ClosureType::formatMangledNameTo(::llvm::raw_ostream &buffer) const {
  buffer << "$Closure<(";
  for (auto [idx, iTy] : llvm::enumerate(getInputTypes())) {
    mlir::reussir::formatMangledNameTo(iTy, buffer);
    if (idx + 1 != getInputTypes().size())
      buffer << ",";
  }
  buffer << "),";
  mlir::reussir::formatMangledNameTo(getOutputType(), buffer);
  buffer << ">";
}

void ReussirDialect::registerTypes() {
  (void)generatedTypePrinter;
  (void)generatedTypeParser;
  // Register tablegen'd types.
  addTypes<CompositeType, UnionType,
#define GET_TYPEDEF_LIST
#include "Reussir/IR/ReussirOpsTypes.cpp.inc"
           >();
}
} // namespace REUSSIR_DECL_SCOPE
} // namespace mlir

namespace mlir {
namespace reussir {

Type ReussirDialect::parseType(DialectAsmParser &parser) const {
  llvm::SMLoc typeLoc = parser.getCurrentLocation();
  StringRef mnemonic;
  Type genType;

  // Try to parse as a tablegen'd type.
  OptionalParseResult parseResult =
      generatedTypeParser(parser, &mnemonic, genType);
  if (parseResult.has_value())
    return genType;

  // Type is not tablegen'd: try to parse as a raw C++ type.
  return StringSwitch<function_ref<Type()>>(mnemonic)
      .Case("composite", [&] { return CompositeType::parse(parser); })
      .Case("union", [&] { return UnionType::parse(parser); })
      .Default([&] {
        parser.emitError(typeLoc) << "unknown reussir type: " << mnemonic;
        return Type();
      })();
}

void ReussirDialect::printType(Type type, DialectAsmPrinter &os) const {
  // Try to print as a tablegen'd type.
  if (generatedTypePrinter(type, os).succeeded())
    return;

  // Type is not tablegen'd: try printing as a raw C++ type.
  TypeSwitch<Type>(type)
      .Case<CompositeType>([&](CompositeType type) {
        os << type.getMnemonic();
        type.print(os);
      })
      .Case<UnionType>([&](UnionType type) {
        os << type.getMnemonic();
        type.print(os);
      })
      .Default([](Type) {
        llvm::report_fatal_error("printer is missing a handler for this type");
      });
}

// ReussirCompositeLayoutInterface
::mlir::reussir::CompositeLayout
RcBoxType::getCompositeLayout(::mlir::DataLayout layout) const {
  auto ptrTy = ::mlir::LLVM::LLVMPointerType::get(getContext());
  auto ptrEqSize =
      ::mlir::IntegerType::get(getContext(), layout.getTypeSizeInBits(ptrTy));
  return getFreezingKind().getValue() != FreezingKind::nonfreezing
             ? CompositeLayout{layout, {ptrEqSize, ptrTy, ptrTy, getDataType()}}
             : CompositeLayout{layout, {ptrEqSize, getDataType()}};
}
::mlir::reussir::CompositeLayout
VectorType::getCompositeLayout(::mlir::DataLayout layout) const {
  auto idxTy = ::mlir::IndexType::get(getContext());
  auto ptrTy = ::mlir::LLVM::LLVMPointerType::get(getContext());
  return {layout, {ptrTy, idxTy, idxTy}};
}
::mlir::reussir::CompositeLayout
OpaqueType::getCompositeLayout(::mlir::DataLayout layout) const {
  auto ptrTy = ::mlir::LLVM::LLVMPointerType::get(getContext());
  auto size = getSize().getUInt();
  auto align = getAlignment().getUInt();
  auto dataTy = ::mlir::LLVM::LLVMFixedVectorType::get(
      ::mlir::IntegerType::get(getContext(), 8), align);
  auto dataArea = ::mlir::LLVM::LLVMArrayType::get(dataTy, size / align);
  return {layout, {ptrTy, ptrTy, dataArea}};
}

::mlir::reussir::CompositeLayout
ClosureType::getCompositeLayout(::mlir::DataLayout layout) const {
  auto ptrTy = mlir::LLVM::LLVMPointerType::get(getContext());
  auto indexTy =
      mlir::IntegerType::get(getContext(), layout.getTypeSizeInBits(ptrTy));
  return {layout, {ptrTy, ptrTy, indexTy}};
}

void populateLLVMTypeConverter(CompositeLayoutCache &cache,
                               mlir::LLVMTypeConverter &converter) {
  converter.addConversion([](RefType type) -> Type {
    return mlir::LLVM::LLVMPointerType::get(type.getContext());
  });
  converter.addConversion([](RcType type) -> Type {
    return mlir::LLVM::LLVMPointerType::get(type.getContext());
  });
  converter.addConversion([](TokenType type) -> Type {
    return mlir::LLVM::LLVMPointerType::get(type.getContext());
  });
  converter.addConversion([](MRefType type) -> Type {
    return mlir::LLVM::LLVMPointerType::get(type.getContext());
  });
  converter.addConversion([](RegionCtxType type) -> Type {
    return mlir::LLVM::LLVMPointerType::get(type.getContext());
  });
  converter.addConversion([](NullableType type) -> Type {
    return mlir::LLVM::LLVMPointerType::get(type.getContext());
  });
  converter.addConversion([](PtrType type) -> Type {
    return mlir::LLVM::LLVMPointerType::get(type.getContext());
  });
  converter.addConversion(
      [&converter, &cache](ReussirCompositeLayoutInterface type) -> Type {
        return cache.get(type).getLLVMType(converter);
      });
  converter.addConversion([&converter](ArrayType type) -> Type {
    auto eltTy = converter.convertType(type.getElementType());
    for (auto size : llvm::reverse(type.getSizes()))
      eltTy = mlir::LLVM::LLVMArrayType::get(eltTy, size);
    return eltTy;
  });
}

namespace detail {
CompositeLayout UnionTypeImpl::getCompositeLayout(mlir::MLIRContext *ctx,
                                                  ::mlir::DataLayout layout,
                                                  ArrayRef<Type> innerTypes) {
  auto tagType = getTagType(ctx, innerTypes);
  auto [dataSz, dataAlign] = getDataLayout(layout, innerTypes);
  auto areaTy = mlir::LLVM::LLVMArrayType::get(mlir::IntegerType::get(ctx, 8),
                                               dataSz.getFixedValue());
  return {layout,
          {tagType},
          CompositeLayout::UnionBody{
              areaTy,
              dataAlign.value(),
          }};
}
LogicalResult
UnionTypeImpl::verify(llvm::function_ref<mlir::InFlightDiagnostic()> emitError,
                      ArrayRef<Type> innerTypes, mlir::StringAttr name,
                      bool incomplete) {
  for (auto type : innerTypes) {
    if (llvm::isa<RefType>(type))
      return emitError() << "cannot have a reference type in a union type";
    if (auto rcTy = llvm::dyn_cast<RcType>(type)) {
      if (rcTy.getFreezingKind().getValue() == FreezingKind::unfrozen)
        return emitError() << "cannot have a non-frozen but freezable RC type "
                              "in a union type, use mref instead";
    }
  }
  if (name && name.getValue().empty()) {
    emitError() << "an identified union type cannot have an empty name";
    return mlir::failure();
  }
  return mlir::success();
}
IntegerType UnionTypeImpl::getTagType(mlir::MLIRContext *ctx,
                                      ArrayRef<Type> innerTypes) {
  return ::mlir::IntegerType::get(
      ctx, ::llvm::Log2_64(::llvm::PowerOf2Ceil(innerTypes.size())));
}
std::pair<::llvm::TypeSize, ::llvm::Align>
UnionTypeImpl::getDataLayout(::mlir::DataLayout layout,
                             ArrayRef<Type> innerTypes) {
  ::llvm::TypeSize size = ::llvm::TypeSize::getFixed(0);
  ::llvm::Align alignment{1};
  for (auto type : innerTypes) {
    size = std::max(size, ::llvm::TypeSize::getFixed(layout.getTypeSize(type)));
    alignment =
        std::max(alignment, ::llvm::Align(layout.getTypeABIAlignment(type)));
  }
  size = ::llvm::alignTo(size, alignment.value());
  return {size, alignment};
}
void UnionTypeImpl::formatMangledNameTo(llvm::raw_ostream &buffer,
                                        ArrayRef<Type> innerTypes,
                                        mlir::StringAttr name) {
  if (name) {
    buffer << name.getValue();
    return;
  }
  buffer << "$Union<";
  for (auto [idx, iTy] : llvm::enumerate(innerTypes)) {
    mlir::reussir::formatMangledNameTo(iTy, buffer);
    if (idx + 1 != innerTypes.size())
      buffer << ",";
  }
  buffer << ">";
}
CompositeLayout
CompositeTypeImpl::getCompositeLayout(mlir::MLIRContext *ctx,
                                      ::mlir::DataLayout layout,
                                      ArrayRef<Type> innerTypes) {
  return {layout, innerTypes};
}
LogicalResult CompositeTypeImpl::verify(
    llvm::function_ref<mlir::InFlightDiagnostic()> emitError,
    ArrayRef<Type> innerTypes, mlir::StringAttr name, bool incomplete) {
  for (auto type : innerTypes) {
    if (llvm::isa<RefType>(type))
      return emitError() << "cannot have a reference type in a composite type";
    if (auto rcTy = llvm::dyn_cast<RcType>(type)) {
      if (rcTy.getFreezingKind().getValue() == FreezingKind::unfrozen)
        return emitError() << "cannot have a non-frozen but freezable RC type "
                              "in a composite type, use mref instead";
    }
  }
  if (name && name.getValue().empty()) {
    emitError() << "an identified composite type cannot have an empty name";
    return mlir::failure();
  }
  return mlir::success();
}
void CompositeTypeImpl::formatMangledNameTo(llvm::raw_ostream &buffer,
                                            ArrayRef<Type> innerTypes,
                                            mlir::StringAttr name) {
  if (name) {
    buffer << name.getValue();
    return;
  }
  buffer << "$Composite<";
  for (auto [idx, iTy] : llvm::enumerate(innerTypes)) {
    mlir::reussir::formatMangledNameTo(iTy, buffer);
    if (idx + 1 != innerTypes.size())
      buffer << ",";
  }
  buffer << ">";
}
template struct ContainerLikeTypeStorage<detail::CompositeTypeImpl>;
template struct ContainerLikeTypeStorage<detail::UnionTypeImpl>;
} // namespace detail
template class ContainerLikeType<detail::CompositeTypeImpl>;
template class ContainerLikeType<detail::UnionTypeImpl>;
} // namespace reussir
} // namespace mlir
