#pragma once

#include "Reussir/Common.h"
#include "Reussir/IR/ReussirDialect.h"
#include "Reussir/IR/ReussirTypes.h"

#define GET_OP_CLASSES
#include "Reussir/IR/ReussirOps.h.inc"

namespace mlir {
namespace REUSSIR_DECL_SCOPE {} // namespace REUSSIR_DECL_SCOPE
} // namespace mlir
