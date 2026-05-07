cmake_minimum_required(VERSION 3.22)
project(face_pipeline VERSION 0.1.0 LANGUAGES CXX)

# ---- Project conventions ----
list(APPEND CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake")
include(CompilerSettings)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

# ---- Options ----
option(BUILD_TOOLS "Build CLI tools (face_enroll, benchmark)" ON)
option(BUILD_TESTS "Build unit tests" OFF)

# ---- Dependencies ----
find_package(CUDAToolkit 12.0 REQUIRED)
find_package(OpenCV 4.5 REQUIRED COMPONENTS core imgproc imgcodecs cudaimgproc cudawarping)
find_package(Eigen3 3.4 REQUIRED NO_MODULE)
find_package(spdlog REQUIRED)
find_package(yaml-cpp REQUIRED)

find_package(PkgConfig REQUIRED)
pkg_check_modules(GST REQUIRED IMPORTED_TARGET
    gstreamer-1.0
    gstreamer-app-1.0
    gobject-2.0
    glib-2.0
)

find_package(FAISS REQUIRED)
find_package(DeepStream 8.0 REQUIRED)
find_package(TensorRT REQUIRED)

# ---- Library: face_pipeline (static) ----
file(GLOB_RECURSE FACE_PIPELINE_SOURCES
    "${CMAKE_SOURCE_DIR}/src/config/*.cpp"
    "${CMAKE_SOURCE_DIR}/src/utils/*.cpp"
    "${CMAKE_SOURCE_DIR}/src/trt/*.cpp"
    "${CMAKE_SOURCE_DIR}/src/align/*.cpp"
    "${CMAKE_SOURCE_DIR}/src/indexing/*.cpp"
    "${CMAKE_SOURCE_DIR}/src/pipeline/*.cpp"
)

add_library(face_pipeline STATIC ${FACE_PIPELINE_SOURCES})
target_include_directories(face_pipeline
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/include>
        ${OpenCV_INCLUDE_DIRS}
        ${TensorRT_INCLUDE_DIRS}
        ${DeepStream_INCLUDE_DIRS}
)
target_link_libraries(face_pipeline
    PUBLIC
        CUDA::cudart
        CUDA::cuda_driver
        ${OpenCV_LIBS}
        Eigen3::Eigen
        spdlog::spdlog
        yaml-cpp::yaml-cpp
        FAISS::FAISS
        TensorRT::nvinfer
        TensorRT::nvonnxparser
        DeepStream::nvds_meta
        DeepStream::nvbufsurface
        PkgConfig::GST
)
set_target_properties(face_pipeline PROPERTIES POSITION_INDEPENDENT_CODE ON)

# ---- Executable: face_server ----
add_executable(face_server src/main.cpp)
target_link_libraries(face_server PRIVATE face_pipeline)

# ---- Tools ----
if(BUILD_TOOLS)
    add_executable(face_enroll tools/face_enroll.cpp)
    target_link_libraries(face_enroll PRIVATE face_pipeline)

    add_executable(face_benchmark tools/benchmark.cpp)
    target_link_libraries(face_benchmark PRIVATE face_pipeline)
endif()

# ---- Tests ----
if(BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()

# ---- Install ----
install(TARGETS face_server RUNTIME DESTINATION bin)
if(BUILD_TOOLS)
    install(TARGETS face_enroll face_benchmark RUNTIME DESTINATION bin)
endif()
install(DIRECTORY configs/ DESTINATION share/face_pipeline/configs)

# ---- Status summary ----
message(STATUS "")
message(STATUS "face_pipeline ${PROJECT_VERSION} configuration:")
message(STATUS "  Build type:   ${CMAKE_BUILD_TYPE}")
message(STATUS "  C++ standard: ${CMAKE_CXX_STANDARD}")
message(STATUS "  CUDA:         ${CUDAToolkit_VERSION}")
message(STATUS "  OpenCV:       ${OpenCV_VERSION}")
message(STATUS "  TensorRT:     ${TensorRT_VERSION}")
message(STATUS "  DeepStream:   ${DeepStream_VERSION}")
message(STATUS "  Build tools:  ${BUILD_TOOLS}")
message(STATUS "  Build tests:  ${BUILD_TESTS}")
message(STATUS "")
