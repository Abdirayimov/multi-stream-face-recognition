include(FetchContent)

# Pinned rather than tracking main: an unannounced GoogleTest change should
# never be able to turn this repository's CI red.
FetchContent_Declare(
    googletest
    GIT_REPOSITORY https://github.com/google/googletest.git
    GIT_TAG        v1.14.0
    GIT_SHALLOW    TRUE
)
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
set(INSTALL_GTEST OFF CACHE BOOL "" FORCE)

# GoogleTest is third-party; this project's warning set is not its contract.
get_directory_property(FACE_PIPELINE_WARNINGS COMPILE_OPTIONS)
set_directory_properties(PROPERTIES COMPILE_OPTIONS "")
FetchContent_MakeAvailable(googletest)
set_directory_properties(PROPERTIES COMPILE_OPTIONS "${FACE_PIPELINE_WARNINGS}")

include(GoogleTest)

set(FACE_PIPELINE_TEST_SOURCES
    test_align.cpp
    test_letterbox.cpp
    test_scrfd_decode.cpp
    test_config.cpp
    test_match_decision.cpp
)

add_executable(face_pipeline_tests ${FACE_PIPELINE_TEST_SOURCES})
target_link_libraries(face_pipeline_tests PRIVATE face_pipeline_core GTest::gtest_main)
target_compile_definitions(face_pipeline_tests
    PRIVATE FACE_PIPELINE_TEST_FIXTURES="${CMAKE_CURRENT_SOURCE_DIR}/fixtures"
)

gtest_discover_tests(face_pipeline_tests)
