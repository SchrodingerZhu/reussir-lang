cmake_minimum_required(VERSION 3.20)
project(reussir-runtime-allocator CXX)
include(FetchContent)
include(CheckCXXCompilerFlag)
FetchContent_Declare(
  snmalloc
  GIT_REPOSITORY https://github.com/microsoft/snmalloc
  GIT_TAG        a3fe420af250f4662aced78b167e45166d202898
)
FetchContent_GetProperties(snmalloc)
if (CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
  check_cxx_compiler_flag("-Wno-missing-template-arg-list-after-template-kw" HAS_NO_MISSING_TEMPLATE_ARG_LIST_AFTER_TEMPLATE_KW)
  if (HAS_NO_MISSING_TEMPLATE_ARG_LIST_AFTER_TEMPLATE_KW)
    add_compile_options("-Wno-missing-template-arg-list-after-template-kw")
  endif()
endif()
if(NOT snmalloc_POPULATED)
  FetchContent_Populate(snmalloc)
  set(SNMALLOC_BUILD_TESTING OFF CACHE BOOL "" FORCE)
  set(SNMALLOC_HEADER_ONLY_LIBRARY ON CACHE BOOL "" FORCE)
  add_subdirectory(${snmalloc_SOURCE_DIR} ${snmalloc_BINARY_DIR})
endif()
file(GLOB_RECURSE REUSE_RUNTIME_SOURCES src/allocator.cpp)
add_library(reussir-allocator STATIC ${REUSE_RUNTIME_SOURCES})
target_link_libraries(reussir-allocator PRIVATE snmalloc)
target_compile_options(reussir-allocator PRIVATE -fno-exceptions -fno-rtti -nostdlib++)

if ("${CMAKE_SYSTEM_NAME}" STREQUAL "FreeBSD")
  target_compile_definitions(reussir-allocator PRIVATE SNMALLOC_USE_THREAD_DESTRUCTORS=1)
elseif (UNIX AND NOT APPLE)
  target_compile_definitions(reussir-allocator PRIVATE SNMALLOC_USE_PTHREAD_DESTRUCTORS=1)
endif()
