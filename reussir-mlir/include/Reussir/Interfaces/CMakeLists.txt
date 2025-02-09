function(add_mlir_type_interface interface)
  set(LLVM_TARGET_DEFINITIONS ${interface}.td)
  mlir_tablegen(${interface}.h.inc -gen-type-interface-decls)
  mlir_tablegen(${interface}.cpp.inc -gen-type-interface-defs)
  add_public_tablegen_target(MLIR${interface}IncGen)
  add_dependencies(mlir-generic-headers MLIR${interface}IncGen)
endfunction()

add_mlir_type_interface(ReussirMangleInterface)
add_mlir_type_interface(ReussirCompositeLayoutInterface)

add_reuseir_mlir_doc(ReussirMangleInterface ReussirMangleInterface
    ./ --gen-type-interface-docs)
add_reuseir_mlir_doc(ReussirCompositeLayoutInterface ReussirCompositeLayoutInterface
    ./ --gen-type-interface-docs)
