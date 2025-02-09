cmake_minimum_required(VERSION 3.22)
project(Reussir-MLIR C CXX)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_FLAGS "-Wno-deprecated-this-capture -Wno-ambiguous-reversed-operator")
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

option(REUSSIR_ENABLE_TESTS "Build tests for Reussir" ON)

include(cmake/FindLLVM.cmake)
include(cmake/FindMLIR.cmake)
message(STATUS "include dirs for llvm and mlir: ${MLIR_INCLUDE_DIRS};${LLVM_INCLUDE_DIRS}")
set(MLIR_TABLEGEN_OUTPUT_DIR ${CMAKE_CURRENT_BINARY_DIR}/include)
message(STATUS "Tablegen output directory: ${MLIR_TABLEGEN_OUTPUT_DIR}")
include_directories(include ${MLIR_INCLUDE_DIRS} ${LLVM_INCLUDE_DIRS} ${MLIR_TABLEGEN_OUTPUT_DIR})

if (REUSSIR_ENABLE_TESTS)
  add_compile_definitions(REUSSIR_ENABLE_TESTS=1)
endif()

add_subdirectory(include)
add_subdirectory(src)

if (REUSSIR_ENABLE_TESTS)
  enable_testing()
  add_subdirectory(test)
endif()
