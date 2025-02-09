add_custom_target(reuseir-doc)

# This replicates part of the add_mlir_doc cmake function from MLIR that cannot
# be used here. This happens because it expects to be run inside MLIR directory
# which is not the case for Reussir.
function(add_reuseir_mlir_doc doc_filename output_file output_directory command)
  set(LLVM_TARGET_DEFINITIONS ${doc_filename}.td)
  tablegen(MLIR ${output_file}.md ${command}
      -dialect reussir
      # The MLIR docs use Hugo, so we allow Hugo specific features here.
      -allow-hugo-specific-features
      ${ARGN}
  )
  set(GEN_DOC_FILE ${PROJECT_BINARY_DIR}/docs/${output_directory}${output_file}.md)
  add_custom_command(
          OUTPUT ${GEN_DOC_FILE}
          COMMAND ${CMAKE_COMMAND} -E copy
                  ${CMAKE_CURRENT_BINARY_DIR}/${output_file}.md
                  ${GEN_DOC_FILE}
          DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/${output_file}.md)
  add_custom_target(${output_file}DocGen DEPENDS ${GEN_DOC_FILE})
  add_dependencies(reuseir-doc ${output_file}DocGen)
endfunction()

add_subdirectory(Interfaces)
add_subdirectory(IR)

set(LLVM_TARGET_DEFINITIONS Passes.td)
mlir_tablegen(Passes.h.inc -gen-pass-decls -name Reussir)
mlir_tablegen(Passes.capi.h.inc -gen-pass-capi-header --prefix Reussir)
mlir_tablegen(Passes.capi.cpp.inc -gen-pass-capi-impl --prefix Reussir)
add_public_tablegen_target(MLIRReussirPassIncGen)

add_reuseir_mlir_doc(Passes ReussirPasses ./ --gen-pass-doc)
