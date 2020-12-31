cmake_minimum_required(VERSION 3.18)
project(gnn-c++ CXX)

if (NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE "Release" CACHE STRING "Select CMake build type." FORCE)

  set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug" "Release")
endif ()

string (TOUPPER "${CMAKE_BUILD_TYPE}" uppercase_CMAKE_BUILD_TYPE)

if (CMAKE_BUILD_TYPE AND NOT uppercase_CMAKE_BUILD_TYPE MATCHES "^(DEBUG|RELEASE)$")
  message (FATAL_ERROR "Invalid value for CMAKE_BUILD_TYPE: ${CMAKE_BUILD_TYPE}")
endif ()

if (CMAKE_CURRENT_SOURCE_DIR STREQUAL CMAKE_CURRENT_BINARY_DIR AND NOT MSVC)
  message(FATAL_ERROR "In-source builds are not allowed.
Please create a directory and run cmake from there, passing the path
to this source directory as the last argument.
This process created the file `CMakeCache.txt' and the directory `CMakeFiles'.
Please delete them.")
endif ()

set(TF_SOURCE_DIR "." CACHE PATH "Source directory to Tensorflow")

find_library(LIB_TFLITE tensorflowlite "${TF_SOURCE_DIR}/bazel-bin/tensorflow/lite/")

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)

add_executable(tflite-strided-slice strided_slice.cc)

target_include_directories(tflite-strided-slice PRIVATE
  ${TF_SOURCE_DIR}
  ${TF_SOURCE_DIR}/tensorflow/lite
  ${TF_SOURCE_DIR}/tensorflow/lite/tools/make/downloads
  ${TF_SOURCE_DIR}/tensorflow/lite/tools/make/downloads/flatbuffers/include
)

if (NOT MSVC)
  set(CXXFLAGS -g -Wall -Wextra -Werror)
  target_compile_definitions(tflite-strided-slice PRIVATE ${CXXDEFS})
  target_compile_options(tflite-strided-slice PRIVATE ${CXXFLAGS}
    -Wno-unused-parameter
  )
  target_link_libraries(tflite-strided-slice PRIVATE ${LDFLAGS} ${LIB_TFLITE})
endif ()
