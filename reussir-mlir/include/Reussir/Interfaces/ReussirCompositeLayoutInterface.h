#pragma once

#include "Reussir/Common.h"
#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMTypes.h"
#include "mlir/IR/Types.h"
#include "mlir/Interfaces/DataLayoutInterfaces.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"
#include <variant>

namespace mlir {
namespace reussir {
class CompositeLayout {
public:
  using FieldKind = std::variant<mlir::Type, size_t>;
  struct Field {
    size_t index;
    size_t byteOffset;
    llvm::Align alignment;
  };
  struct UnionBody {
    LLVM::LLVMArrayType dataArea;
    size_t alignment;
  };

private:
  llvm::Align alignment = llvm::Align{1};
  llvm::TypeSize size = llvm::TypeSize::getFixed(0);
  llvm::SmallVector<FieldKind> raw_fields;
  llvm::DenseMap<size_t, Field> field_map;

public:
  const llvm::Align &getAlignment() const { return alignment; }
  const llvm::TypeSize &getSize() const { return size; }
  llvm::ArrayRef<FieldKind> getRawFields() const { return raw_fields; }
  Field getField(size_t idx) const { return field_map.at(idx); }

  CompositeLayout(mlir::DataLayout layout, llvm::ArrayRef<mlir::Type> fields,
                  std::optional<UnionBody> unionBody = std::nullopt);

  mlir::LLVM::LLVMStructType
  getLLVMType(const mlir::LLVMTypeConverter &converter) const;
};
} // namespace reussir
} // namespace mlir

#include "Reussir/Interfaces/ReussirCompositeLayoutInterface.h.inc"

namespace mlir {
namespace reussir {
class CompositeLayoutCache {
  mlir::DataLayout dataLayout;
  llvm::DenseMap<ReussirCompositeLayoutInterface, CompositeLayout> cache;

public:
  CompositeLayoutCache(mlir::DataLayout dataLayout)
      : dataLayout(dataLayout), cache{} {}
  const CompositeLayout &get(const ReussirCompositeLayoutInterface &iface);
  mlir::DataLayout getDataLayout() const { return dataLayout; }
};
} // namespace reussir
} // namespace mlir
