# FindAdept.cmake
#
# Finds the Adept library
#
# This will define the following variables
#
#    ADEPT_FOUND
#    ADEPT_LIBRARIES
#    ADEPT_INCLUDE_DIRS
#

find_path(ADEPT_INCLUDE_DIR adept.h PATHS "${ADEPT_DIR}/include")
find_library(ADEPT_LIBRARY NAMES libadept.a adept PATHS "${ADEPT_DIR}/lib")

set(ADEPT_INCLUDE_DIRS "${ADEPT_INCLUDE_DIR}")
set(ADEPT_LIBRARIES "${ADEPT_LIBRARY}")

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(Adept
   "Must specify Adept installation directory in cmake config file (-DADEPT_DIR=\"path/to/adept\")"
   ADEPT_INCLUDE_DIR ADEPT_LIBRARY
)

mark_as_advanced(ADEPT_INCLUDE_DIR ADEPT_LIBRARY)
