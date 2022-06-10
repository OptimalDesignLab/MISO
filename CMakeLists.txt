cmake_minimum_required(VERSION 3.13...3.15)

# set the project name, version, description, and languages
project(ODLMach VERSION 1.0
                DESCRIPTION "Optimal Design Lab multi-physics solver"
                LANGUAGES CXX)

include(GNUInstallDirs)
include(CMakePackageConfigHelpers)
                
if (CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)
   set(CMAKE_INSTALL_PREFIX "${CMAKE_SOURCE_DIR}" CACHE PATH "default source dir" FORCE)
   message("CMAKE_INSTALL_PREFIX = ${CMAKE_INSTALL_PREFIX}")
endif (CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)

# Load Mach cmake utilities
include(cmake/project-is-top-level.cmake)
include(cmake/mach_cmake_utils.cmake)

# Let CMake know where to find custom FindFoo.cmake files
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake/find")

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
find_package(MFEM CONFIG REQUIRED)

# configure_file(
#   "${PROJECT_SOURCE_DIR}/cmake/config.hpp.in"
#   "${PROJECT_BINARY_DIR}/_config.hpp")

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

# find the MPI compilers
find_package(MPI REQUIRED)

cmake_policy(SET CMP0077 NEW)
set(TINYSPLINE_ENABLE_CXX ON)
set(TINYSPLINE_INSTALL ON)
include(FetchContent)
FetchContent_Declare(tinyspline
   GIT_REPOSITORY "https://github.com/tuckerbabcock/tinyspline.git"
   # GIT_REPOSITORY "https://github.com/msteinbeck/tinyspline.git"
)

FetchContent_MakeAvailable(tinyspline)
set_target_properties(tinysplinecxx
   PROPERTIES 
      POSITION_INDEPENDENT_CODE ON
)

set(JSON_Install ON)
FetchContent_Declare(json
   URL https://github.com/nlohmann/json/releases/download/v3.10.5/json.tar.xz
)

FetchContent_MakeAvailable(json)

# FetchContent_Declare(json
#   GIT_REPOSITORY https://github.com/nlohmann/json.git
#   GIT_TAG v3.10.2)

# FetchContent_GetProperties(json)
# if(NOT json_POPULATED)
#   FetchContent_Populate(json)
#   add_subdirectory(${json_SOURCE_DIR} ${json_BINARY_DIR})
# endif()

# create build target library named mach
add_library(mach "")

# add_public_headers_to_target(mach "mach.hpp")

set_target_properties(mach PROPERTIES PUBLIC_HEADER "")

# set(MACH_HEADERS "mach.hpp")

# target_sources(mach
#    PRIVATE
#       "${MACH_HEADERS}"
# )

# Use CMakeLists.txt files in subdirectories to add sources to mach
add_subdirectory(src)

target_link_libraries(mach
   PUBLIC
      Adept::adept
      mfem
      # "${PUMI_LIBRARIES}" # shouldn't need to link since MFEM handles it
      nlohmann_json::nlohmann_json
   PRIVATE
      tinysplinecxx
)

target_include_directories(mach
   PUBLIC
      "${MFEM_INCLUDE_DIRS}"
      "${PUMI_INCLUDE_DIRS}"
)

# if (MFEM_USE_EGADS)
#    target_link_libraries(mach
#       PUBLIC
#          "${EGADS_LIBRARIES}"
#    )

#    target_include_directories(mach
#       PUBLIC
#          "${EGADS_INCLUDE_DIRS}"
#    )
# endif (MFEM_USE_EGADS)

target_compile_features(mach
   PUBLIC
      cxx_std_17
)

set_target_properties(mach
   PROPERTIES 
      CXX_EXTENSIONS OFF
      POSITION_INDEPENDENT_CODE ON
      # CXX_VISIBILITY_PRESET hidden
      # # VISIBILITY_INLINES_HIDDEN True
      # # INTERPROCEDURAL_OPTIMIZATION True
)

option(MACH_USE_CLANG_TIDY
      "Use clang-tidy static analysis tool"
      NO)
if (MACH_USE_CLANG_TIDY)
   set_target_properties(mach
      PROPERTIES
         CXX_CLANG_TIDY "clang-tidy;--format-style=file;--extra-arg=--std=c++17"
         # CXX_CLANG_TIDY "clang-tidy;--fix;--fix-errors;--format-style=file;--extra-arg=--std=c++11"
   )
endif (MACH_USE_CLANG_TIDY)

set(DEBUG_OPTIONS
   "-g"
   -Wall
   # -Wextra
   # -Werror
)

set(RELEASE_OPTIONS
   "-O2"
)

target_compile_options(mach
   PRIVATE 
      "$<$<CONFIG:DEBUG>:${DEBUG_OPTIONS}>"
      "$<$<CONFIG:RELEASE>:${RELEASE_OPTIONS}>"
)

# TODO: When EGADS support is in mainline PUMI move this to MFEM CMake
if (MFEM_USE_EGADS)
   target_compile_definitions(mach
      PUBLIC
         MFEM_USE_EGADS
   )
endif (MFEM_USE_EGADS)

install(
   TARGETS mach
   EXPORT mach-targets
   LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
   ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
   RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
   INCLUDES DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
   PUBLIC_HEADER DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)

configure_package_config_file(
   "${CMAKE_CURRENT_SOURCE_DIR}/config.cmake.in"
   "${CMAKE_CURRENT_BINARY_DIR}/mach-config.cmake"
   INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/mach
)

write_basic_package_version_file(
   "${PROJECT_BINARY_DIR}/mach-config-version.cmake"
   COMPATIBILITY AnyNewerVersion
)

install(FILES
   "${PROJECT_BINARY_DIR}/mach-config.cmake"
   "${PROJECT_BINARY_DIR}/mach-config-version.cmake"
   DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/mach
)

install(
   EXPORT mach-targets
   NAMESPACE mach::
   DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/mach
)

get_target_property(MACH_SOURCES mach SOURCES)
add_custom_target(mach-format
   COMMAND clang-format
   -style=file
   -i
   ${MACH_SOURCES}
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
      "Build a python wrapper for mach that allows constructing solvers and
      running problems through a Python interface"
      NO)

if (BUILD_PYTHON_WRAPPER)

   # TODO: maybe don't do this? Use c++14/17 features if they're available on the compiler?
   # set(PYBIND11_CPP_STANDARD -std=c++17)

   include(FetchContent)

   cmake_policy(SET CMP0127 NEW)
   FetchContent_Declare(pybind11
      GIT_REPOSITORY "https://github.com/pybind/pybind11"
      GIT_TAG v2.7.1
   )

   FetchContent_MakeAvailable(pybind11)

   # FetchContent_Declare(pybind11_json
   #    GIT_REPOSITORY "https://github.com/pybind/pybind11_json"
   # )

   # FetchContent_MakeAvailable(pybind11_json)

   pybind11_add_module(pyMach "")

   target_compile_features(pyMach
   PUBLIC
      cxx_std_17
   )

   if (MACH_USE_CLANG_TIDY)
      set_target_properties(pyMach
         PROPERTIES
            CXX_CLANG_TIDY "clang-tidy;--format-style=file;--extra-arg=--std=c++17"
            # CXX_CLANG_TIDY "clang-tidy;--fix;--fix-errors;--format-style=file;--extra-arg=--std=c++11"
      )
   endif (MACH_USE_CLANG_TIDY)

   add_subdirectory(mach)

   get_target_property(PYMACH_SOURCES pyMach SOURCES)
   add_custom_target(pymach-format
      COMMAND clang-format
      -style=file
      -i
      ${PYMACH_SOURCES}
   )

   if (BUILD_TESTING)
      target_compile_definitions(pyMach
         PRIVATE
            BUILD_TESTING
      )

      target_sources(pyMach
         PRIVATE
            "${CMAKE_CURRENT_SOURCE_DIR}/test/regression/test_mach_inputs.hpp"
      )
      target_include_directories(pyMach
         PRIVATE
            "${CMAKE_CURRENT_SOURCE_DIR}/test/regression/"
      )

   endif (BUILD_TESTING)

   target_link_libraries(pyMach
      PUBLIC
         mach
         # "${MFEM_LIBRARIES}"
   )

   if (MFEM_USE_EGADS)
      target_link_libraries(pyMach
         PUBLIC
            "${EGADS_LIBRARIES}"
      )

      target_include_directories(pyMach
         PUBLIC
            "${EGADS_INCLUDE_DIRS}"
      )
   endif (MFEM_USE_EGADS)

   # find the mpi4py c interface include directories and include them
   execute_process(COMMAND
      python -c "import mpi4py; print(mpi4py.get_include(), end ='')"
      OUTPUT_VARIABLE MPI4PY_INCLUDE_DIRS
   )
   
   target_include_directories(pyMach
      PRIVATE
         "${MPI4PY_INCLUDE_DIRS}"
   )

   # target_compile_options(pyMach
   #    PRIVATE 
   #       "$<$<CONFIG:DEBUG>:${DEBUG_OPTIONS}>"
   #       "$<$<CONFIG:RELEASE>:${RELEASE_OPTIONS}>"
   # )
   # set_target_properties(pyMach
   #    PROPERTIES 
   #       CXX_EXTENSIONS OFF
   #       POSITION_INDEPENDENT_CODE ON
   #       CXX_VISIBILITY_PRESET default
   #       VISIBILITY_INLINES_HIDDEN True
   #       INTERPROCEDURAL_OPTIMIZATION OFF
   # )

   install(TARGETS pyMach
      LIBRARY DESTINATION "${CMAKE_SOURCE_DIR}/mach"
   )

endif (BUILD_PYTHON_WRAPPER)

add_custom_target(format
   DEPENDS mach-format pymach-format
)
