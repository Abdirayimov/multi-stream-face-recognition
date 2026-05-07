# FindFAISS.cmake
#
# Locates the FAISS GPU library (https://github.com/facebookresearch/faiss).
#
# Defines:
#   FAISS_FOUND          - True if FAISS was found
#   FAISS_INCLUDE_DIRS   - Include directories for FAISS headers
#   FAISS_LIBRARIES      - The FAISS library to link against
#   FAISS::FAISS         - Imported target

find_path(FAISS_INCLUDE_DIR
    NAMES faiss/Index.h
    HINTS
        /usr/local/include
        /usr/include
        ${FAISS_ROOT}/include
)

find_library(FAISS_LIBRARY
    NAMES faiss
    HINTS
        /usr/local/lib
        /usr/lib/x86_64-linux-gnu
        ${FAISS_ROOT}/lib
)

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(FAISS
    REQUIRED_VARS FAISS_LIBRARY FAISS_INCLUDE_DIR
)

if(FAISS_FOUND AND NOT TARGET FAISS::FAISS)
    add_library(FAISS::FAISS UNKNOWN IMPORTED)
    set_target_properties(FAISS::FAISS PROPERTIES
        IMPORTED_LOCATION "${FAISS_LIBRARY}"
        INTERFACE_INCLUDE_DIRECTORIES "${FAISS_INCLUDE_DIR}"
    )
    set(FAISS_INCLUDE_DIRS "${FAISS_INCLUDE_DIR}")
    set(FAISS_LIBRARIES "${FAISS_LIBRARY}")
endif()

mark_as_advanced(FAISS_INCLUDE_DIR FAISS_LIBRARY)
