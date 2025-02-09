get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
add_mlir_dialect_library(MLIRReussir
  STATIC
    ReussirDialect.cpp
    ReussirTypes.cpp
    ReussirAttrs.cpp
    ReussirOps.cpp
    ReussirOpsEnums.cpp
  DEPENDS
    MLIRReussirInterfaces
    MLIRReussirOpsIncGen
    MLIRReussirEnumsIncGen
  LINK_COMPONENTS
    Core
  LINK_LIBS PUBLIC
    ${dialect_libs}
    MLIRReussirInterfaces
)
