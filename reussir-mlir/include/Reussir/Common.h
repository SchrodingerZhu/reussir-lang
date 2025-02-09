#pragma once
#include "llvm/ADT/StringRef.h"
#include <algorithm>
#include <array>
#if defined(__GNUG__) && __has_attribute(visibility)
#define REUSSIR_DECL_SCOPE [[gnu::visibility("hidden")]] reussir
#else
#define REUSSIR_DECL_SCOPE reussir
#endif

#if __has_include("mlir/Support/LogicalResult.h")
#include "mlir/Support/LogicalResult.h"
namespace mlir::reussir {
using mlir::LogicalResult;
inline LogicalResult success() { return LogicalResult::success(); }
} // namespace mlir::reussir
#else
#include "llvm/Support/LogicalResult.h"
namespace mlir::reussir {
using llvm::LogicalResult;
inline LogicalResult success() { return LogicalResult::success(); }
} // namespace mlir::reussir
#endif

namespace mlir::reussir {
template <size_t N> struct StringLiteral {
  constexpr StringLiteral(const char (&str)[N]) {
    std::copy_n(str, N, value.begin());
  }
  std::array<char, N> value;
  constexpr operator llvm::StringRef() const { return {&value[0], N - 1}; }
};

template <size_t N> StringLiteral(const char (&str)[N]) -> StringLiteral<N>;

template <StringLiteral Str> constexpr inline decltype(Str) operator""_str() {
  return Str;
}

} // namespace mlir::reussir
