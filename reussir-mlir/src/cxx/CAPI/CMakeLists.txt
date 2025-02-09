add_mlir_dialect_library(MLIRReussirCAPI
  STATIC
    CAPI.cpp
  DEPENDS
    MLIRReussir
    MLIRReussirAnalysis
    MLIRReussirPasses
    MLIRReussirInterfaces
  LINK_COMPONENTS
    Core
  LINK_LIBS PUBLIC
    MLIRReussir
    MLIRReussirAnalysis
    MLIRReussirPasses
    MLIRReussirInterfaces
)
