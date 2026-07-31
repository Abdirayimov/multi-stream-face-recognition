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
option(BUILD_DEEPSTREAM "Build the DeepStream pipeline (requires DeepStream SDK + GStreamer)" ON)
option(BUILD_FAISS "Build the FAISS-based searcher (requires FAISS GPU)" ON)
option(FACE_PIPELINE_BUILD_TESTS "Build the GoogleTest unit test suite" OFF)
option(FACE_PIPELINE_CPU_ONLY
       "Build only the CPU-only algorithmic core (no CUDA / TensorRT / FAISS / DeepStream)" OFF)

# Accepted for backwards compatibility with pre-0.2 build scripts.
if(BUILD_TESTS AND NOT FACE_PIPELINE_BUILD_TESTS)
    message(WARNING "BUILD_TESTS is deprecated; use FACE_PIPELINE_BUILD_TESTS")
    set(FACE_PIPELINE_BUILD_TESTS ON)
endif()

if(FACE_PIPELINE_CPU_ONLY)
    # The GPU stages have no meaning without their SDKs; turn them off
    # rather than letting find_package fail late.
    set(BUILD_TOOLS OFF)
    set(BUILD_DEEPSTREAM OFF)
    set(BUILD_FAISS OFF)
endif()

# ---- Dependencies ----
# The algorithmic core needs only these three; everything else is a GPU
# stage and is looked up separately below.
find_package(OpenCV 4.5 REQUIRED COMPONENTS core imgproc)
find_package(Eigen3 3.4 REQUIRED NO_MODULE)
find_package(yaml-cpp REQUIRED)

if(NOT FACE_PIPELINE_CPU_ONLY)
    find_package(CUDAToolkit 12.0 REQUIRED)
    find_package(OpenCV 4.5 REQUIRED COMPONENTS core imgproc imgcodecs videoio)
    find_package(spdlog REQUIRED)
    find_package(TensorRT REQUIRED)
endif()

if(BUILD_FAISS)
    find_package(FAISS)
    if(NOT FAISS_FOUND)
        message(WARNING "FAISS not found; disabling FAISS searcher")
        set(BUILD_FAISS OFF)
    endif()
endif()

if(BUILD_DEEPSTREAM)
    find_package(PkgConfig REQUIRED)
    pkg_check_modules(GST REQUIRED IMPORTED_TARGET
        gstreamer-1.0
        gstreamer-app-1.0
        gobject-2.0
        glib-2.0
    )
    find_package(DeepStream 8.0)
    if(NOT DeepStream_FOUND)
        message(WARNING "DeepStream SDK not found; disabling DeepStream pipeline")
        set(BUILD_DEEPSTREAM OFF)
    endif()
endif()

# ---- Library: face_pipeline_core (static, CPU only) ----
# Config parsing, the Umeyama aligner and the SCRFD decoder are pure CPU
# math. Keeping them in their own target lets the unit tests — and CI
# runners without an NVIDIA stack — build them without CUDA or TensorRT.
add_library(face_pipeline_core STATIC
    src/config/system_config.cpp
    src/align/similarity_transform.cpp
    src/align/face_aligner.cpp
    src/trt/scrfd_postprocess.cpp
)
target_include_directories(face_pipeline_core
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/include>
        ${OpenCV_INCLUDE_DIRS}
)
target_link_libraries(face_pipeline_core
    PUBLIC
        opencv_core
        opencv_imgproc
        Eigen3::Eigen
        yaml-cpp::yaml-cpp
)
set_target_properties(face_pipeline_core PROPERTIES POSITION_INDEPENDENT_CODE ON)

# ---- Library: face_pipeline (static, GPU stages) ----
if(NOT FACE_PIPELINE_CPU_ONLY)
    file(GLOB_RECURSE FACE_PIPELINE_SOURCES
        "${CMAKE_SOURCE_DIR}/src/utils/*.cpp"
        "${CMAKE_SOURCE_DIR}/src/trt/*.cpp"
        "${CMAKE_SOURCE_DIR}/src/indexing/*.cpp"
        "${CMAKE_SOURCE_DIR}/src/pipeline/*.cpp"
    )
    # Already compiled into face_pipeline_core.
    list(FILTER FACE_PIPELINE_SOURCES EXCLUDE REGEX "scrfd_postprocess\\.cpp$")

    if(NOT BUILD_FAISS)
        list(FILTER FACE_PIPELINE_SOURCES EXCLUDE REGEX "faiss_searcher\\.cpp$")
        list(FILTER FACE_PIPELINE_SOURCES EXCLUDE REGEX "probe_chain\\.cpp$")
    endif()
    if(NOT BUILD_DEEPSTREAM)
        list(FILTER FACE_PIPELINE_SOURCES EXCLUDE REGEX "deepstream_pipeline\\.cpp$")
    endif()

    add_library(face_pipeline STATIC ${FACE_PIPELINE_SOURCES})
    target_include_directories(face_pipeline
        PUBLIC
            $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/include>
            ${OpenCV_INCLUDE_DIRS}
            ${TensorRT_INCLUDE_DIRS}
    )
    target_link_libraries(face_pipeline
        PUBLIC
            face_pipeline_core
            CUDA::cudart
            CUDA::cuda_driver
            ${OpenCV_LIBS}
            spdlog::spdlog
            TensorRT::nvinfer
            TensorRT::nvonnxparser
    )
    if(BUILD_FAISS)
        target_link_libraries(face_pipeline PUBLIC FAISS::FAISS)
        target_compile_definitions(face_pipeline PUBLIC FACE_PIPELINE_HAVE_FAISS=1)
    endif()
    if(BUILD_DEEPSTREAM)
        target_include_directories(face_pipeline PUBLIC ${DeepStream_INCLUDE_DIRS})
        target_link_libraries(face_pipeline PUBLIC
            DeepStream::nvds_meta
            DeepStream::nvbufsurface
            PkgConfig::GST
        )
        target_compile_definitions(face_pipeline PUBLIC FACE_PIPELINE_HAVE_DEEPSTREAM=1)
    endif()
    set_target_properties(face_pipeline PROPERTIES POSITION_INDEPENDENT_CODE ON)
endif()

# ---- Executable: face_server ----
# The full server wires the DeepStream pipeline to the FAISS searcher,
# so it only builds when both are available.
if(BUILD_FAISS AND BUILD_DEEPSTREAM)
    add_executable(face_server src/main.cpp)
    target_link_libraries(face_server PRIVATE face_pipeline)
endif()

# ---- Tools ----
if(BUILD_TOOLS)
    # face_detect needs only the SCRFD detector, so it builds in every
    # configuration (incl. detection-only builds without FAISS/DeepStream).
    add_executable(face_detect tools/face_detect.cpp)
    target_link_libraries(face_detect PRIVATE face_pipeline)

    # enroll + benchmark exercise the FAISS index.
    if(BUILD_FAISS)
        add_executable(face_enroll tools/face_enroll.cpp)
        target_link_libraries(face_enroll PRIVATE face_pipeline)

        add_executable(face_benchmark tools/benchmark.cpp)
        target_link_libraries(face_benchmark PRIVATE face_pipeline)
    endif()
endif()

# ---- Tests ----
if(FACE_PIPELINE_BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()

# ---- Install ----
if(TARGET face_server)
    install(TARGETS face_server RUNTIME DESTINATION bin)
endif()
if(TARGET face_detect)
    install(TARGETS face_detect RUNTIME DESTINATION bin)
endif()
if(TARGET face_enroll)
    install(TARGETS face_enroll face_benchmark RUNTIME DESTINATION bin)
endif()
install(DIRECTORY configs/ DESTINATION share/face_pipeline/configs)

# ---- Status summary ----
message(STATUS "")
message(STATUS "face_pipeline ${PROJECT_VERSION} configuration:")
message(STATUS "  Build type:        ${CMAKE_BUILD_TYPE}")
message(STATUS "  C++ standard:      ${CMAKE_CXX_STANDARD}")
message(STATUS "  CPU-only:          ${FACE_PIPELINE_CPU_ONLY}")
if(NOT FACE_PIPELINE_CPU_ONLY)
    message(STATUS "  CUDA:              ${CUDAToolkit_VERSION}")
    message(STATUS "  TensorRT:          ${TensorRT_VERSION}")
endif()
message(STATUS "  OpenCV:            ${OpenCV_VERSION}")
message(STATUS "  FAISS path:        ${BUILD_FAISS}")
message(STATUS "  DeepStream path:   ${BUILD_DEEPSTREAM}")
message(STATUS "  Build tools:       ${BUILD_TOOLS}")
message(STATUS "  Build tests:       ${FACE_PIPELINE_BUILD_TESTS}")
message(STATUS "")
