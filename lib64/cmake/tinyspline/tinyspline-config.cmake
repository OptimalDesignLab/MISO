
####### Expanded from @PACKAGE_INIT@ by configure_package_config_file() #######
####### Any changes to this file will be overwritten by the next CMake run ####
####### The input file was tinyspline-config.cmake.in                            ########

get_filename_component(PACKAGE_PREFIX_DIR "${CMAKE_CURRENT_LIST_DIR}/../../../" ABSOLUTE)

macro(set_and_check _var _file)
  set(${_var} "${_file}")
  if(NOT EXISTS "${_file}")
    message(FATAL_ERROR "File or directory ${_file} referenced by variable ${_var} does not exist !")
  endif()
endmacro()

macro(check_required_components _NAME)
  foreach(comp ${${_NAME}_FIND_COMPONENTS})
    if(NOT ${_NAME}_${comp}_FOUND)
      if(${_NAME}_FIND_REQUIRED_${comp})
        set(${_NAME}_FOUND FALSE)
      endif()
    endif()
  endforeach()
endmacro()

####################################################################################
set_and_check(TINYSPLINE_INCLUDE_DIRS "${PACKAGE_PREFIX_DIR}/include")
set_and_check(TINYSPLINE_LIBRARY_DIRS "${PACKAGE_PREFIX_DIR}/lib64")
set(TINYSPLINE_BINARY_DIRS "${PACKAGE_PREFIX_DIR}/bin")
set(TINYSPLINE_VERSION "0.4.0")
set(TINYSPLINE_DEFINITIONS "")

include("${CMAKE_CURRENT_LIST_DIR}/tinyspline-targets.cmake")
set(TINYSPLINE_LIBRARIES tinyspline::tinyspline)

mark_as_advanced(
	TINYSPLINE_INCLUDE_DIRS
	TINYSPLINE_LIBRARY_DIRS
	TINYSPLINE_BINARY_DIRS
	TINYSPLINE_VERSION
	TINYSPLINE_DEFINITIONS
	TINYSPLINE_LIBRARIES)

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(tinyspline
	REQUIRED_VARS
		TINYSPLINE_INCLUDE_DIRS
		TINYSPLINE_LIBRARY_DIRS
		TINYSPLINE_BINARY_DIRS
		TINYSPLINE_LIBRARIES
	VERSION_VAR
		TINYSPLINE_VERSION)
