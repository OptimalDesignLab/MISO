#----------------------------------------------------------------
# Generated CMake target import file for configuration "Release".
#----------------------------------------------------------------

# Commands may need to know the format version.
set(CMAKE_IMPORT_FILE_VERSION 1)

# Import target "tinyspline::tinyspline" for configuration "Release"
set_property(TARGET tinyspline::tinyspline APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(tinyspline::tinyspline PROPERTIES
  IMPORTED_LINK_INTERFACE_LANGUAGES_RELEASE "C"
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/lib64/libtinyspline.a"
  )

list(APPEND _cmake_import_check_targets tinyspline::tinyspline )
list(APPEND _cmake_import_check_files_for_tinyspline::tinyspline "${_IMPORT_PREFIX}/lib64/libtinyspline.a" )

# Commands beyond this point should not need to know the version.
set(CMAKE_IMPORT_FILE_VERSION)
