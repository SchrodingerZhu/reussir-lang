add_mlir_dialect(ReussirOps reussir)

set(LLVM_TARGET_DEFINITIONS ReussirOps.td)
mlir_tablegen(ReussirOpsEnums.h.inc -gen-enum-decls)
mlir_tablegen(ReussirOpsEnums.cpp.inc -gen-enum-defs)
mlir_tablegen(ReussirOpsStructs.h.inc -gen-attrdef-decls)
mlir_tablegen(ReussirOpsStructs.cpp.inc -gen-attrdef-defs)
mlir_tablegen(ReussirOpsAttributes.h.inc -gen-attrdef-decls)
mlir_tablegen(ReussirOpsAttributes.cpp.inc -gen-attrdef-defs)
add_public_tablegen_target(MLIRReussirEnumsIncGen)
add_dependencies(mlir-headers MLIRReussirEnumsIncGen)

add_reuseir_mlir_doc(ReussirOps ReussirOps ./ --gen-op-doc)
# FIXME: uncomment the following line once we have added an attribute.
# add_reuseir_mlir_doc(ReussirAttrs ReussirAttrs ./ --gen-attrdef-doc)
add_reuseir_mlir_doc(ReussirTypes ReussirTypes ./ --gen-typedef-doc)
add_reuseir_mlir_doc(ReussirDialect ReussirDialect ./ --gen-dialect-doc)
