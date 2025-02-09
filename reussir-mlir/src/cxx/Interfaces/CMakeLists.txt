get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(conversion_libs GLOBAL PROPERTY MLIR_CONVERSION_LIBS)
add_mlir_dialect_library(MLIRReussirInterfaces
  STATIC
    ReussirMangleInterface.cpp
    ReussirCompositeLayoutInterface.cpp
  DEPENDS
    MLIRReussirMangleInterfaceIncGen
    MLIRReussirCompositeLayoutInterfaceIncGen
  LINK_COMPONENTS
    Core
  LINK_LIBS PUBLIC
    ${dialect_libs}
    ${conversion_libs}
    MLIRIR
    MLIRSupport
)
