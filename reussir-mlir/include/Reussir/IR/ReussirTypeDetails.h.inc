#include "mlir/IR/MLIRContext.h"
namespace detail {

template <typename Impl> struct ContainerLikeTypeStorage : public TypeStorage {
  struct KeyTy {
    ArrayRef<Type> innerTypes;
    StringAttr name;
    bool incomplete;

    KeyTy(ArrayRef<Type> innerTypes, StringAttr name, bool incomplete)
        : innerTypes(innerTypes), name(name), incomplete(incomplete) {}
  };

  ArrayRef<Type> innerTypes;
  StringAttr name;
  bool incomplete;

  ContainerLikeTypeStorage(ArrayRef<Type> innerTypes, StringAttr name,
                           bool incomplete)
      : innerTypes(innerTypes), name(name), incomplete(incomplete) {}

  KeyTy getAsKey() const { return KeyTy(innerTypes, name, incomplete); }

  bool operator==(const KeyTy &key) const {
    if (name)
      return name == key.name;
    return (innerTypes == key.innerTypes) && (name == key.name) &&
           (incomplete == key.incomplete);
  }

  static llvm::hash_code hashKey(const KeyTy &key) {
    if (key.name)
      return llvm::hash_combine(Impl::name, key.name);
    return llvm::hash_combine(key.innerTypes, key.incomplete);
  }

  static ContainerLikeTypeStorage *construct(TypeStorageAllocator &allocator,
                                             const KeyTy &key) {
    return new (allocator.allocate<ContainerLikeTypeStorage>())
        ContainerLikeTypeStorage(allocator.copyInto(key.innerTypes), key.name,
                                 key.incomplete);
  }

  /// Mutates the innerTypes and attributes an identified struct.
  ///
  /// Once a record is mutated, it is marked as complete, preventing further
  /// mutations. Anonymous structs are always complete and cannot be mutated.
  /// This method does not fail if a mutation of a complete struct does not
  /// change the struct.
  LogicalResult mutate(TypeStorageAllocator &allocator,
                       ArrayRef<Type> innerTypes) {
    // Anonymous structs cannot mutate.
    if (!name)
      return failure();

    // Mutation of complete structs are allowed if they change nothing.
    if (!incomplete)
      return mlir::success(this->innerTypes == innerTypes);

    // Mutate incomplete struct.
    this->innerTypes = allocator.copyInto(innerTypes);

    incomplete = false;
    return success();
  }
};

} // namespace detail

template <typename TypeImpl>
Type ContainerLikeType<TypeImpl>::parse(mlir::AsmParser &parser) {
  FailureOr<AsmParser::CyclicParseReset> cyclicParseGuard;
  const auto loc = parser.getCurrentLocation();
  const auto eLoc = parser.getEncodedSourceLoc(loc);
  auto *context = parser.getContext();

  if (parser.parseLess())
    return {};

  mlir::StringAttr name;
  parser.parseOptionalAttribute(name);

  // Is a self reference: ensure referenced type was parsed.
  if (name && parser.parseOptionalGreater().succeeded()) {
    auto type = getChecked(eLoc, context, name);
    if (succeeded(parser.tryStartCyclicParse(type))) {
      parser.emitError(loc, "invalid self-reference within composite type");
      return {};
    }
    return type;
  }

  // Is a named composite definition: ensure name has not been parsed yet.
  if (name) {
    auto type = getChecked(eLoc, context, name);
    cyclicParseGuard = parser.tryStartCyclicParse(type);
    if (failed(cyclicParseGuard)) {
      parser.emitError(loc, "composite already defined");
      return {};
    }
  }

  // Parse record innerTypes or lack thereof.
  bool incomplete = true;
  llvm::SmallVector<mlir::Type> innerTypes;
  if (parser.parseOptionalKeyword("incomplete").failed()) {
    incomplete = false;
    const auto delimiter = AsmParser::Delimiter::Braces;
    const auto parseElementFn = [&parser, &innerTypes]() {
      return parser.parseType(innerTypes.emplace_back());
    };
    if (parser.parseCommaSeparatedList(delimiter, parseElementFn).failed())
      return {};
  }

  if (parser.parseGreater())
    return {};

  // Try to create the proper record type.
  ArrayRef<mlir::Type> innerTypesRef(
      innerTypes); // Needed for template deduction.
  mlir::Type type = {};
  if (name && incomplete) { // Identified & incomplete
    type = getChecked(eLoc, context, name);
  } else if (name && !incomplete) { // Identified & complete
    type = getChecked(eLoc, context, innerTypesRef, name);
    // If the record has a self-reference, its type already exists in a
    // incomplete state. In this case, we must complete it.
    if (mlir::cast<ContainerLikeType>(type).isIncomplete())
      mlir::cast<ContainerLikeType>(type).complete(innerTypesRef);
  } else if (!name && !incomplete) { // anonymous & complete
    type = getChecked(eLoc, context, innerTypesRef);
  } else { // anonymous & incomplete
    parser.emitError(loc, "anonymous composite types must be complete");
    return {};
  }

  return type;
}

template <typename TypeImpl>
void ContainerLikeType<TypeImpl>::print(mlir::AsmPrinter &printer) const {
  FailureOr<AsmPrinter::CyclicPrintReset> cyclicPrintGuard;
  printer << '<';

  if (getName())
    printer << getName();

  // Current type has already been printed: print as self reference.
  cyclicPrintGuard = printer.tryStartCyclicPrint(*this);
  if (failed(cyclicPrintGuard)) {
    printer << '>';
    return;
  }

  // Type not yet printed: continue printing the entire record.
  if (getName())
    printer << ' ';

  if (isIncomplete()) {
    printer << "incomplete";
  } else {
    printer << "{";
    llvm::interleaveComma(getInnerTypes(), printer);
    printer << "}";
  }

  printer << '>';
}

template <typename TypeImpl>
ContainerLikeType<TypeImpl>
ContainerLikeType<TypeImpl>::get(::mlir::MLIRContext *context,
                                 ArrayRef<Type> innerTypes, StringAttr name) {
  return Base::get(context, innerTypes, name, false);
}

template <typename TypeImpl>
ContainerLikeType<TypeImpl> ContainerLikeType<TypeImpl>::getChecked(
    ::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError,
    ::mlir::MLIRContext *context, ArrayRef<Type> innerTypes, StringAttr name) {
  return Base::getChecked(emitError, context, innerTypes, name,
                          /*incomplete=*/false);
}

template <typename TypeImpl>
ContainerLikeType<TypeImpl>
ContainerLikeType<TypeImpl>::get(::mlir::MLIRContext *context,
                                 StringAttr name) {
  return Base::get(context, /*innerTypes=*/ArrayRef<Type>{}, name,
                   /*incomplete=*/true);
}

template <typename TypeImpl>
ContainerLikeType<TypeImpl> ContainerLikeType<TypeImpl>::getChecked(
    ::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError,
    ::mlir::MLIRContext *context, StringAttr name) {
  return Base::getChecked(emitError, context, ArrayRef<Type>{}, name,
                          /*incomplete=*/true);
}

template <typename TypeImpl>
ContainerLikeType<TypeImpl>
ContainerLikeType<TypeImpl>::get(::mlir::MLIRContext *context,
                                 ArrayRef<Type> innerTypes) {
  return Base::get(context, innerTypes, StringAttr{}, /*incomplete=*/false);
}

template <typename TypeImpl>
ContainerLikeType<TypeImpl> ContainerLikeType<TypeImpl>::getChecked(
    ::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError,
    ::mlir::MLIRContext *context, ArrayRef<Type> innerTypes) {
  return Base::getChecked(emitError, context, innerTypes, StringAttr{},
                          /*incomplete=*/false);
}

template <typename TypeImpl>
::llvm::ArrayRef<mlir::Type>
ContainerLikeType<TypeImpl>::getInnerTypes() const {
  return Base::getImpl()->innerTypes;
}

template <typename TypeImpl>
bool ContainerLikeType<TypeImpl>::isIncomplete() const {
  return Base::getImpl()->incomplete;
}

template <typename TypeImpl>
mlir::StringAttr ContainerLikeType<TypeImpl>::getName() const {
  return Base::getImpl()->name;
}

template <typename TypeImpl>
bool ContainerLikeType<TypeImpl>::getIncomplete() const {
  return Base::getImpl()->incomplete;
}

template <typename TypeImpl>
void ContainerLikeType<TypeImpl>::complete(ArrayRef<Type> innerTypes) {
  if (Base::mutate(innerTypes).failed())
    llvm_unreachable("failed to complete struct");
}
namespace detail {
struct CompositeTypeImpl {
  static constexpr llvm::StringLiteral name = "reussir.composite";
  static constexpr llvm::StringLiteral getMnemonic() { return "composite"; }
  static ::mlir::reussir::CompositeLayout
  getCompositeLayout(mlir::MLIRContext *, ::mlir::DataLayout layout,
                     ArrayRef<Type> innerTypes);
  static ::mlir::LogicalResult
  verify(llvm::function_ref<mlir::InFlightDiagnostic()> emitError,
         ArrayRef<Type> innerTypes, mlir::StringAttr name, bool incomplete);
  static void formatMangledNameTo(llvm::raw_ostream &buffer,
                                  ArrayRef<Type> innerTypes,
                                  mlir::StringAttr name);
};
struct UnionTypeImpl {
  static constexpr llvm::StringLiteral name = "reussir.union";
  static constexpr llvm::StringLiteral getMnemonic() { return "union"; }
  static ::mlir::reussir::CompositeLayout
  getCompositeLayout(mlir::MLIRContext *ctx, ::mlir::DataLayout layout,
                     ArrayRef<Type> innerTypes);
  static ::mlir::LogicalResult
  verify(llvm::function_ref<mlir::InFlightDiagnostic()> emitError,
         ArrayRef<Type> innerTypes, mlir::StringAttr name, bool incomplete);
  static ::mlir::IntegerType getTagType(mlir::MLIRContext *ctx,
                                        ArrayRef<Type> innerTypes);
  static std::pair<::llvm::TypeSize, ::llvm::Align>
  getDataLayout(::mlir::DataLayout layout, ArrayRef<Type> innerTypes);
  static void formatMangledNameTo(llvm::raw_ostream &buffer,
                                  ArrayRef<Type> innerTypes,
                                  mlir::StringAttr name);
};

extern template struct ContainerLikeTypeStorage<detail::CompositeTypeImpl>;
extern template struct ContainerLikeTypeStorage<detail::UnionTypeImpl>;

} // namespace detail

extern template class ContainerLikeType<detail::CompositeTypeImpl>;
extern template class ContainerLikeType<detail::UnionTypeImpl>;
