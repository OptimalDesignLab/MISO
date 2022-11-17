
####### Expanded from @PACKAGE_INIT@ by configure_package_config_file() #######
####### Any changes to this file will be overwritten by the next CMake run ####
####### The input file was tinysplinecxx-config.cmake.in                            ########

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
set_and_check(TINYSPLINECXX_INCLUDE_DIRS "${PACKAGE_PREFIX_DIR}/include")
set_and_check(TINYSPLINECXX_LIBRARY_DIRS "${PACKAGE_PREFIX_DIR}/lib64")
set(TINYSPLINECXX_BINARY_DIRS "${PACKAGE_PREFIX_DIR}/bin")
set(TINYSPLINECXX_VERSION "0.4.0")
set(TINYSPLINECXX_DEFINITIONS "")

include("${CMAKE_CURRENT_LIST_DIR}/tinysplinecxx-targets.cmake")
set(TINYSPLINECXX_LIBRARIES tinysplinecxx::tinysplinecxx)

mark_as_advanced(
	TINYSPLINECXX_INCLUDE_DIRS
	TINYSPLINECXX_LIBRARY_DIRS
	TINYSPLINECXX_BINARY_DIRS
	TINYSPLINECXX_VERSION
	TINYSPLINECXX_DEFINITIONS
	TINYSPLINECXX_LIBRARIES)

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(tinysplinecxx
	REQUIRED_VARS
		TINYSPLINECXX_INCLUDE_DIRS
		TINYSPLINECXX_LIBRARY_DIRS
		TINYSPLINECXX_BINARY_DIRS
		TINYSPLINECXX_LIBRARIES
	VERSION_VAR
		TINYSPLINE_VERSION)
