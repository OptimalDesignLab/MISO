cmake_minimum_required(VERSION 3.13...3.15)

# set the project name, version, description, and languages
project(ODLMach VERSION 1.0
                DESCRIPTION "Optimal Design Lab multi-physics solver"
                LANGUAGES CXX)

find_package(MFEM CONFIG REQUIRED PATHS "${MFEM_DIR}" NO_DEFAULT_PATH)


# if (MFEM_USE_CUDA)
#    # MFEM_USE_CUDA requires CMake 3.8 or newer (for direct CUDA support)
#    cmake_minimum_required(VERSION 3.8 FATAL_ERROR)
#    enable_language(CUDA)
#    message(STATUS "Using CUDA architecture: ${CUDA_ARCH}")
# endif()


# find_package(Git QUIET)
# if(GIT_FOUND AND EXISTS "${PROJECT_SOURCE_DIR}/.git")
# # Update submodules as needed
#     option(GIT_SUBMODULE "Check submodules during build" ON)
#     if(GIT_SUBMODULE)
#         message(STATUS "Submodule update")
#         execute_process(COMMAND ${GIT_EXECUTABLE} submodule update --init --recursive
#                         WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
#                         RESULT_VARIABLE GIT_SUBMOD_RESULT)
#         if(NOT GIT_SUBMOD_RESULT EQUAL "0")
#             message(FATAL_ERROR "git submodule update --init failed with ${GIT_SUBMOD_RESULT}, please checkout submodules")
#         endif()
#     endif()
# endif()

# if(NOT EXISTS "${PROJECT_SOURCE_DIR}/extern/mfem/CMakeLists.txt")
#     message(FATAL_ERROR "The submodules were not downloaded! GIT_SUBMODULE was turned off or failed. Please update submodules and try again.")
# endif()

# create build target library named mach, will be static library
add_library(mach STATIC "")

# Use CMakeLists.txt files in subdirectories to add sources to mach
add_subdirectory(src)
# # mfem git submodule can be added directly since it has a CMakeLists.txt file
# add_subdirectory(extern/mfem)


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
   INCLUDES DESTINATION include)


