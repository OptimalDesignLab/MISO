cmake_minimum_required(VERSION 3.13...3.15)

# set the project name, version, description, and languages
project(ODLMISO VERSION 1.0
                DESCRIPTION "Optimal Design Lab multi-physics solver"
                LANGUAGES CXX)

if (CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)
   set(CMAKE_INSTALL_PREFIX "${CMAKE_SOURCE_DIR}" CACHE PATH "default source dir" FORCE)
   message("CMAKE_INSTALL_PREFIX = ${CMAKE_INSTALL_PREFIX}")
endif (CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)

# Let CMake know where to find custom FindFoo.cmake files
set(CMAKE_MODULE_PATH
   ${CMAKE_MODULE_PATH} "${CMAKE_CURRENT_SOURCE_DIR}/cmake/")

### correctly handle rpath when installing
### From: https://gitlab.kitware.com/cmake/community/-/wikis/doc/cmake/RPATH-handling
# use, i.e. don't skip the full RPATH for the build tree
set(CMAKE_SKIP_BUILD_RPATH FALSE)

# when building, don't use the install RPATH already
# (but later on when installing)
set(CMAKE_BUILD_WITH_INSTALL_RPATH FALSE)

set(CMAKE_INSTALL_RPATH "${CMAKE_INSTALL_PREFIX}/lib")

# add the automatically determined parts of the RPATH
# which point to directories outside the build tree to the install RPATH
set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)

# the RPATH to be used when installing, but only if it's not a system directory
list(FIND CMAKE_PLATFORM_IMPLICIT_LINK_DIRECTORIES "${CMAKE_INSTALL_PREFIX}/lib" isSystemDir)
if("${isSystemDir}" STREQUAL "-1")
    set(CMAKE_INSTALL_RPATH "${CMAKE_INSTALL_PREFIX}/lib")
endif("${isSystemDir}" STREQUAL "-1")
### end of install rpath stuff

find_package(Adept MODULE REQUIRED)
find_package(MFEM CONFIG REQUIRED PATHS "${MFEM_DIR}" NO_DEFAULT_PATH)

configure_file(
  "${PROJECT_SOURCE_DIR}/cmake/config.hpp.in"
  "${PROJECT_BINARY_DIR}/_config.hpp")

# if MFEM was compiled with PUMI, we need to find it so we can 
if (MFEM_USE_PUMI)
   # If PUMI_DIR was specified, only link to that directory,
   # i.e. don't link to another installation in /usr/lib by mistake
   find_package(SCOREC 2.1.0 REQUIRED OPTIONAL_COMPONENTS gmi_sim gmi_egads
      CONFIG PATHS ${PUMI_DIR} NO_DEFAULT_PATH)
   if (SCOREC_FOUND)
      set(PUMI_FOUND ${SCOREC_FOUND})

      # TODO: when EGADS is in mainline PUMI put this in MFEM CMake
      # Define a header file with the MFEM_USE_EGADS preprocessor variable
      set(MFEM_USE_EGADS ${SCOREC_gmi_egads_FOUND})

      get_target_property(PUMI_INCLUDE_DIRS
         SCOREC::apf INTERFACE_INCLUDE_DIRECTORIES)
      set(PUMI_LIBRARIES SCOREC::core)
   endif (SCOREC_FOUND)
endif (MFEM_USE_PUMI)

# find the MPI compilers used by MFEM
find_package(MPI REQUIRED)

# option(MFEM_USE_EGADS "Build with EGADS support" OFF)
# message(STATUS "MFEM_USE_EGADS: ${MFEM_USE_EGADS}")

# EGADS Support for mesh movement
if (MFEM_USE_EGADS)
   message(STATUS "finding EGADS")
   find_package(EGADS MODULE REQUIRED)
endif (MFEM_USE_EGADS)

# Try to find OpenMP (used by Adept)
# find_package(OpenMP)

include(FetchContent)
FetchContent_Declare(tinyspline
   GIT_REPOSITORY "https://github.com/tuckerbabcock/tinyspline"
)

set(TINYSPLINE_BUILD_EXAMPLES OFF)
set(TINYSPLINE_BUILD_TESTS OFF)
set(TINYSPLINE_BUILD_DOCS OFF)

FetchContent_MakeAvailable(tinyspline)
set_target_properties(tinysplinecxx
   PROPERTIES 
      POSITION_INDEPENDENT_CODE ON
)

FetchContent_Declare(json
  GIT_REPOSITORY https://github.com/nlohmann/json.git
  GIT_TAG v3.10.2)

FetchContent_GetProperties(json)
if(NOT json_POPULATED)
  FetchContent_Populate(json)
  add_subdirectory(${json_SOURCE_DIR} ${json_BINARY_DIR} EXCLUDE_FROM_ALL)
endif()

# create build target library named miso, will be static library
add_library(miso STATIC "")

target_sources(miso
   PRIVATE
      "${CMAKE_CURRENT_SOURCE_DIR}/miso.hpp"
)

# Use CMakeLists.txt files in subdirectories to add sources to miso
add_subdirectory(src)

target_link_libraries(miso
   PUBLIC
      "${ADEPT_LIBRARIES}"
      "${MFEM_LIBRARIES}"
      # "${PUMI_LIBRARIES}" # shouldn't need to link since MFEM handles it
      tinysplinecxx
      nlohmann_json::nlohmann_json
)

target_include_directories(miso
   PUBLIC
      "${ADEPT_INCLUDE_DIRS}"
      "${MFEM_INCLUDE_DIRS}"
      "${PUMI_INCLUDE_DIRS}"
)

if (MFEM_USE_EGADS)
   target_link_libraries(miso
      PUBLIC
         "${EGADS_LIBRARIES}"
   )

   target_include_directories(miso
      PUBLIC
         "${EGADS_INCLUDE_DIRS}"
   )
endif (MFEM_USE_EGADS)

target_compile_features(miso
   PUBLIC
      cxx_std_17
)

set_target_properties(miso
   PROPERTIES 
      CXX_EXTENSIONS OFF
      POSITION_INDEPENDENT_CODE ON
      # CXX_VISIBILITY_PRESET hidden
      # # VISIBILITY_INLINES_HIDDEN True
      # # INTERPROCEDURAL_OPTIMIZATION True
)

option(MISO_USE_CLANG_TIDY
      "Use clang-tidy static analysis tool"
      NO)
if (MISO_USE_CLANG_TIDY)
   set_target_properties(miso
      PROPERTIES
         CXX_CLANG_TIDY "clang-tidy;--format-style=file;--extra-arg=--std=c++11"
         # CXX_CLANG_TIDY "clang-tidy;--fix;--fix-errors;--format-style=file;--extra-arg=--std=c++11"
   )
endif (MISO_USE_CLANG_TIDY)

set(DEBUG_OPTIONS
   "-g"
   -Wall
   # -Wextra
   # -Werror
)

set(RELEASE_OPTIONS
   "-O2"
)

target_compile_options(miso
   PRIVATE 
      "$<$<CONFIG:DEBUG>:${DEBUG_OPTIONS}>"
      "$<$<CONFIG:RELEASE>:${RELEASE_OPTIONS}>"
)

# TODO: When EGADS support is in mainline PUMI move this to MFEM CMake
if (MFEM_USE_EGADS)
   target_compile_definitions(miso
      PUBLIC
         MFEM_USE_EGADS
   )
endif (MFEM_USE_EGADS)

install(TARGETS miso
   OPTIONAL
   EXPORT misoTargets
   LIBRARY DESTINATION lib
   INCLUDES DESTINATION include
)

install(DIRECTORY src
   OPTIONAL
   DESTINATION include
   FILES_MATCHING PATTERN "*.h*"
)

install(FILES miso.hpp
   OPTIONAL
   DESTINATION include
)

get_target_property(MISO_SOURCES miso SOURCES)
add_custom_target(miso-format
   COMMAND clang-format
   -style=file
   -i
   ${MISO_SOURCES}
)

if (CMAKE_PROJECT_NAME STREQUAL PROJECT_NAME)
   include(CTest)
endif ()

if (CMAKE_PROJECT_NAME STREQUAL PROJECT_NAME AND BUILD_TESTING)
   # `make test` only runs the tests by default, doesn't build them
   # work around to use `make tests` to build and run the tests
   add_custom_target(build_tests)
   # 'tests' triggers 'ctest'.
   add_custom_target(tests COMMAND ${CMAKE_CTEST_COMMAND})
   # 'tests' implies 'build_tests'
   add_dependencies(tests build_tests)
   add_subdirectory(test EXCLUDE_FROM_ALL)

   # add_custom_target(verbose_tests COMMAND "${CMAKE_CTEST_COMMAND} -V")
   # # 'verbose_tests' implies 'build_tests'
   # add_dependencies(verbose_tests build_tests)
   # add_subdirectory(test EXCLUDE_FROM_ALL)
endif ()

# create sandbox target
add_custom_target(sandbox)
add_subdirectory(sandbox EXCLUDE_FROM_ALL)

# TODO: This doesn't really work...I used MFEM's approach but should look at scorec's
# add documentation subdirectory
# creates custom target `doc' to build the doxygen documentation
add_subdirectory(doc)

option(BUILD_PYTHON_WRAPPER
      "Build a python wrapper for miso that allows constructing solvers and
      running problems through a Python interface"
      NO)

if (BUILD_PYTHON_WRAPPER)

   # TODO: maybe don't do this? Use c++14/17 features if they're available on the compiler?
   # set(PYBIND11_CPP_STANDARD -std=c++11)

   include(FetchContent)

   FetchContent_Declare(pybind11
      GIT_REPOSITORY "https://github.com/pybind/pybind11"
      GIT_TAG v2.7.1
   )

   FetchContent_MakeAvailable(pybind11)

   # FetchContent_Declare(pybind11_json
   #    GIT_REPOSITORY "https://github.com/pybind/pybind11_json"
   # )

   # FetchContent_MakeAvailable(pybind11_json)

   pybind11_add_module(pyMISO "")

   target_compile_features(pyMISO
   PUBLIC
      cxx_std_17
   )

   if (MISO_USE_CLANG_TIDY)
      set_target_properties(pyMISO
         PROPERTIES
            CXX_CLANG_TIDY "clang-tidy;--format-style=file;--extra-arg=--std=c++11"
            # CXX_CLANG_TIDY "clang-tidy;--fix;--fix-errors;--format-style=file;--extra-arg=--std=c++11"
      )
   endif (MISO_USE_CLANG_TIDY)

   add_subdirectory(miso)

   get_target_property(PYMISO_SOURCES pyMISO SOURCES)
   add_custom_target(pymiso-format
      COMMAND clang-format
      -style=file
      -i
      ${PYMISO_SOURCES}
   )

   if (BUILD_TESTING)
      target_compile_definitions(pyMISO
         PRIVATE
            BUILD_TESTING
      )

      target_sources(pyMISO
         PRIVATE
            "${CMAKE_CURRENT_SOURCE_DIR}/test/regression/test_miso_inputs.hpp"
      )
      target_include_directories(pyMISO
         PRIVATE
            "${CMAKE_CURRENT_SOURCE_DIR}/test/regression/"
      )

   endif (BUILD_TESTING)

   target_link_libraries(pyMISO
      PUBLIC
         miso
         # "${MFEM_LIBRARIES}"
   )

   if (MFEM_USE_EGADS)
      target_link_libraries(pyMISO
         PUBLIC
            "${EGADS_LIBRARIES}"
      )

      target_include_directories(pyMISO
         PUBLIC
            "${EGADS_INCLUDE_DIRS}"
      )
   endif (MFEM_USE_EGADS)

   # find the mpi4py c interface include directories and include them
   execute_process(COMMAND
      python3 -c "import mpi4py; print(mpi4py.get_include(), end ='')"
      OUTPUT_VARIABLE MPI4PY_INCLUDE_DIRS
   )
   
   target_include_directories(pyMISO
      PRIVATE
         "${MPI4PY_INCLUDE_DIRS}"
   )

   # target_compile_options(pyMISO
   #    PRIVATE 
   #       "$<$<CONFIG:DEBUG>:${DEBUG_OPTIONS}>"
   #       "$<$<CONFIG:RELEASE>:${RELEASE_OPTIONS}>"
   # )
   # set_target_properties(pyMISO
   #    PROPERTIES 
   #       CXX_EXTENSIONS OFF
   #       POSITION_INDEPENDENT_CODE ON
   #       CXX_VISIBILITY_PRESET default
   #       VISIBILITY_INLINES_HIDDEN True
   #       INTERPROCEDURAL_OPTIMIZATION OFF
   # )

   install(TARGETS pyMISO
      LIBRARY DESTINATION "${CMAKE_SOURCE_DIR}/miso"
   )

endif (BUILD_PYTHON_WRAPPER)

add_custom_target(format
   DEPENDS miso-format pymiso-format
)
