if (MACH_USE_EGADS)

option(EGADS_LITE "Enable EGADSlite" OFF)
# this file brings EGADS_LITE from CMake to C++
# configure_file("${CMAKE_CURRENT_SOURCE_DIR}/mach_egads_config.hpp.in"
#               "${CMAKE_CURRENT_BINARY_DIR}/mach_egads_config.hpp")

target_sources(mach
   PRIVATE
      mach_egads.cpp
      mach_egads.hpp
)

#add_library(mach_egads ${SOURCES})

target_link_libraries(mach PUBLIC ${EGADS_LIBRARIES})

# Include directories
target_include_directories(mach 
    PUBLIC
        "${CMAKE_CURRENT_SOURCE_DIR}"
        "${EGADS_INCLUDE_DIR})"
)

# make sure the compiler can find the config header
#target_include_directories(mach PRIVATE
#  $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}>)


# mach_export_library(mach_egads)

endif (MACH_USE_EGADS)