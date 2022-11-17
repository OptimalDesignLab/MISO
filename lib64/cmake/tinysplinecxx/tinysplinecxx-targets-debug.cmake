#----------------------------------------------------------------
# Generated CMake target import file for configuration "Debug".
#----------------------------------------------------------------

# Commands may need to know the format version.
set(CMAKE_IMPORT_FILE_VERSION 1)

# Import target "tinysplinecxx::tinysplinecxx" for configuration "Debug"
set_property(TARGET tinysplinecxx::tinysplinecxx APPEND PROPERTY IMPORTED_CONFIGURATIONS DEBUG)
set_target_properties(tinysplinecxx::tinysplinecxx PROPERTIES
  IMPORTED_LINK_INTERFACE_LANGUAGES_DEBUG "C;CXX"
  IMPORTED_LOCATION_DEBUG "${_IMPORT_PREFIX}/lib64/libtinysplinecxx.a"
  )

list(APPEND _cmake_import_check_targets tinysplinecxx::tinysplinecxx )
list(APPEND _cmake_import_check_files_for_tinysplinecxx::tinysplinecxx "${_IMPORT_PREFIX}/lib64/libtinysplinecxx.a" )

# Commands beyond this point should not need to know the version.
set(CMAKE_IMPORT_FILE_VERSION)
