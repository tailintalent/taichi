# ./taichi/runtime/program_impls/metal/CMakeLists.txt

add_library(metal_program_impl)
target_sources(metal_program_impl
  PRIVATE
    metal_program.cpp
  )

target_include_directories(metal_program_impl
  PRIVATE
    ${PROJECT_SOURCE_DIR}
    ${PROJECT_SOURCE_DIR}/external/eigen
    ${PROJECT_SOURCE_DIR}/external/spdlog/include
    ${LLVM_INCLUDE_DIRS}
  )

target_link_libraries(metal_program_impl PRIVATE metal_codegen)
target_link_libraries(metal_program_impl PRIVATE metal_runtime)
