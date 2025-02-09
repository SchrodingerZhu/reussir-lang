add_mlir_dialect_library(MLIRReussirAnalysis
  STATIC
    AliasAnalysis.cpp
    ReuseAnalysis.cpp
  DEPENDS
    MLIRReussir
  LINK_COMPONENTS
    Core
  LINK_LIBS PUBLIC
    MLIRReussir
)
