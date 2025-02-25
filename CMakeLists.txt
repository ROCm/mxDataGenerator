cmake_minimum_required(VERSION 3.16)
set(CMAKE_CXX_STANDARD 20)
project(FPDataGeneration)

include(FetchContent)
FetchContent_Declare(
  googletest
  URL https://github.com/google/googletest/archive/refs/tags/release-1.12.1.zip
)
FetchContent_MakeAvailable(googletest)

enable_testing()

add_compile_options(-Wall -Wextra -Wpedantic -Werror -Wno-strict-aliasing -Wno-error=strict-aliasing -fsanitize=undefined)
add_link_options(-fsanitize=undefined)

find_package(OpenMP)
add_library(mxDataGenerator INTERFACE)
target_link_libraries(mxDataGenerator INTERFACE OpenMP::OpenMP_CXX)
target_include_directories(mxDataGenerator INTERFACE lib/include)

add_executable(
  mxDataGeneratorTests

  test/data_generator_test.cpp
  test/f32_test.cpp
  test/fp16_test.cpp
  test/bf16_test.cpp
  test/ocp_e4m3_mxfp8_test.cpp
  test/ocp_e5m2_mxfp8_test.cpp
  test/ocp_e2m3_mxfp6_test.cpp
  test/ocp_e3m2_mxfp6_test.cpp
  test/ocp_e2m1_mxfp4_test.cpp
)

target_link_libraries(
  mxDataGeneratorTests PRIVATE
  GTest::gtest_main
  mxDataGenerator
  OpenMP::OpenMP_CXX
)

include(GoogleTest)
gtest_discover_tests(mxDataGeneratorTests DISCOVERY_MODE PRE_TEST)
