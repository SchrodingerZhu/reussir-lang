#include "Reussir/Interfaces/ReussirCompositeLayoutInterface.h"

namespace mlir {
namespace reussir {
CompositeLayout::CompositeLayout(mlir::DataLayout layout,
                                 llvm::ArrayRef<mlir::Type> fields,
                                 std::optional<UnionBody> unionBody)
    : alignment(1), size(0, false), raw_fields{}, field_map{} {
  for (auto [index, type] : llvm::enumerate(fields)) {
    llvm::TypeSize typeSz = layout.getTypeSize(type);
    // skip zero sized element
    if (typeSz.isZero())
      continue;
    size_t typeAlign = layout.getTypeABIAlignment(type);
    alignment = std::max(alignment, llvm::Align(typeAlign));
    llvm::TypeSize alignedSize = llvm::alignTo(size, typeAlign);
    if (alignedSize > size)
      raw_fields.emplace_back(alignedSize - size);
    field_map.insert(
        {index, {raw_fields.size(), alignedSize, llvm::Align{typeAlign}}});
    raw_fields.emplace_back(type);
    size = alignedSize + typeSz;
  }
  if (unionBody) {
    llvm::TypeSize typeSz = layout.getTypeSize(unionBody->dataArea);
    size_t typeAlign = unionBody->alignment;
    alignment = std::max(alignment, llvm::Align(typeAlign));
    llvm::TypeSize alignedSize = llvm::alignTo(size, typeAlign);
    if (alignedSize > size)
      raw_fields.emplace_back(alignedSize - size);
    field_map.insert(
        {fields.size(),
         {raw_fields.size(), alignedSize, llvm::Align{typeAlign}}});
    raw_fields.emplace_back(unionBody->dataArea);
    size = alignedSize + typeSz;
  }
  llvm::TypeSize alignedSize = llvm::alignTo(size, alignment.value());
  if (alignedSize > size)
    raw_fields.emplace_back(alignedSize - size);
  size = alignedSize;
}

mlir::LLVM::LLVMStructType
CompositeLayout::getLLVMType(const mlir::LLVMTypeConverter &converter) const {
  llvm::SmallVector<mlir::Type> convertedTypes;
  std::transform(
      raw_fields.begin(), raw_fields.end(), std::back_inserter(convertedTypes),
      [&](const FieldKind &kind) -> mlir::Type {
        return std::visit(
            [&](auto &&variant) -> mlir::Type {
              using T = std::decay_t<decltype(variant)>;
              if constexpr (std::is_same_v<T, mlir::Type>)
                return converter.convertType(variant);
              if constexpr (std::is_same_v<T, size_t>)
                return mlir::LLVM::LLVMArrayType::get(
                    mlir::IntegerType::get(&converter.getContext(), 8),
                    variant);
              llvm_unreachable("conversion logic error");
            },
            kind);
      });
  return mlir::LLVM::LLVMStructType::getLiteral(&converter.getContext(),
                                                convertedTypes);
}

const CompositeLayout &
CompositeLayoutCache::get(const ReussirCompositeLayoutInterface &iface) {
  if (auto it = cache.find(iface); it != cache.end())
    return it->second;
  auto layout = iface.getCompositeLayout(this->dataLayout);
  return cache.try_emplace(iface, layout).first->second;
}
} // namespace reussir
} // namespace mlir

#include "Reussir/Interfaces/ReussirCompositeLayoutInterface.cpp.inc"
