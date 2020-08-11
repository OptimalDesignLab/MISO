cmake_minimum_required(VERSION 3.13...3.15)

# set the project name, version, description, and languages
project(ODLMach VERSION 1.0
                DESCRIPTION "Optimal Design Lab multi-physics solver"
                LANGUAGES CXX)

if (CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)
   set(CMAKE_INSTALL_PREFIX "${CMAKE_SOURCE_DIR}" CACHE PATH "default source dir" FORCE)
   message("CMAKE_INSTALL_PREFIX = ${CMAKE_INSTALL_PREFIX}")
endif (CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)

# Let CMake know where to find custom FindFoo.cmake files
set(CMAKE_MODULE_PATH
   ${CMAKE_MODULE_PATH} "${CMAKE_CURRENT_SOURCE_DIR}/cmake/")

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

option(MACH_USE_EGADS "Build with EGADS support" OFF)
message(STATUS "MACH_USE_EGADS: ${MACH_USE_EGADS}")

# EGADS Support for mesh movement
if (MACH_USE_EGADS)
   message(STATUS "hello")
   find_package(EGADS MODULE REQUIRED)
endif (MACH_USE_EGADS)

# Try to find OpenMP (used by Adept)
# find_package(OpenMP)

# create build target library named mach, will be static library
add_library(mach STATIC "")

target_sources(mach
   PUBLIC

   PRIVATE
      mach.hpp
)

# Use CMakeLists.txt files in subdirectories to add sources to mach
add_subdirectory(src)

target_link_libraries(mach
   PUBLIC
      "${ADEPT_LIBRARIES}"
      "${MFEM_LIBRARIES}"
      # "${PUMI_LIBRARIES}" # shouldn't need to link since MFEM handles it
)

if (MACH_USE_EGADS)
   target_link_libraries(mach
      PUBLIC
         "${EGADS_LIBRARIES}"
   )
endif (MACH_USE_EGADS)

target_include_directories(mach
   PUBLIC
      "${ADEPT_INCLUDE_DIRS}"
      "${MFEM_INCLUDE_DIRS}"
      "${PUMI_INCLUDE_DIRS}"
)

if (MACH_USE_EGADS)
   target_include_directories(mach
      PUBLIC
         "${EGADS_INCLUDE_DIRS}"
   )
endif (MACH_USE_EGADS)

target_compile_features(mach
   PUBLIC
      cxx_std_11
)

set_target_properties(mach
   PROPERTIES 
      CXX_EXTENSIONS OFF
      POSITION_INDEPENDENT_CODE ON
      # CXX_VISIBILITY_PRESET hidden
      # # VISIBILITY_INLINES_HIDDEN True
      # # INTERPROCEDURAL_OPTIMIZATION True
)

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

install(TARGETS mach
   OPTIONAL
   EXPORT machTargets
   LIBRARY DESTINATION lib
   INCLUDES DESTINATION include
)

install(DIRECTORY src
   OPTIONAL
   DESTINATION include
   FILES_MATCHING PATTERN "*.h*"
)

install(FILES mach.hpp
   OPTIONAL
   DESTINATION include
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
   # set(PYBIND11_CPP_STANDARD -std=c++11)

   include(FetchContent)

   FetchContent_Declare(pybind11
      GIT_REPOSITORY "https://github.com/pybind/pybind11"
   )

   FetchContent_MakeAvailable(pybind11)

   # FetchContent_Declare(pybind11_json
   #    GIT_REPOSITORY "https://github.com/pybind/pybind11_json"
   # )

   # FetchContent_MakeAvailable(pybind11_json)

   pybind11_add_module(pyMach "")

   add_subdirectory(mach)

   target_link_libraries(pyMach
      PUBLIC
         mach
         # "${MFEM_LIBRARIES}"
   )

   # find the mpi4py c interface include directories and include them
   execute_process(COMMAND
      python3 -c "import mpi4py; print(mpi4py.get_include(), end ='')"
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

