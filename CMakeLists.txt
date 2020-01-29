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

if (MFEM_USE_PUMI)
   # If PUMI_DIR was specified, only link to that directory,
   # i.e. don't link to another installation in /usr/lib by mistake
   find_package(SCOREC 2.1.0 REQUIRED OPTIONAL_COMPONENTS gmi_sim
      CONFIG PATHS ${PUMI_DIR} NO_DEFAULT_PATH)
   if (SCOREC_FOUND)
      set(PUMI_FOUND ${SCOREC_FOUND})
      get_target_property(PUMI_INCLUDE_DIRS
         SCOREC::apf INTERFACE_INCLUDE_DIRECTORIES)
      set(PUMI_LIBRARIES SCOREC::core)
   endif (SCOREC_FOUND)
endif (MFEM_USE_PUMI)

# create build target library named mach, will be static library
add_library(mach STATIC "")

# Use CMakeLists.txt files in subdirectories to add sources to mach
add_subdirectory(src)

target_link_libraries(mach
   PUBLIC
      "${ADEPT_LIBRARIES}"
      "${MFEM_LIBRARIES}"
      # "${PUMI_LIBRARIES}" # shouldn't need to link since MFEM handles it
)

target_include_directories(mach
   PUBLIC
      "${ADEPT_INCLUDE_DIRS}"
      "${MFEM_INCLUDE_DIRS}"
      "${PUMI_INCLUDE_DIRS}"
)

target_compile_features(mach
   PUBLIC
      cxx_std_11
)

set_target_properties(mach
   PROPERTIES 
      CXX_EXTENSIONS OFF
      POSITION_INDEPENDENT_CODE ON
)

target_compile_options(mach
   PRIVATE 
      -Wall 
      -Wextra
      # -Werror
)

install(TARGETS mach
   EXPORT machTargets
   LIBRARY DESTINATION lib
   INCLUDES DESTINATION include
)

install(DIRECTORY src 
   DESTINATION include
   FILES_MATCHING PATTERN "*.h*"
)


if (CMAKE_PROJECT_NAME STREQUAL PROJECT_NAME)
   include(CTest)
endif ()

if (CMAKE_PROJECT_NAME STREQUAL PROJECT_NAME AND BUILD_TESTING)
   add_subdirectory(test)
endif ()
