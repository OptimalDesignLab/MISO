if (MFEM_USE_EGADS)

option(EGADS_LITE "Enable EGADSlite" OFF)
# this file brings EGADS_LITE from CMake to C++
# configure_file("${CMAKE_CURRENT_SOURCE_DIR}/mach_egads_config.hpp.in"
#               "${CMAKE_CURRENT_BINARY_DIR}/mach_egads_config.hpp")

set(MACH_UTILITY_EGADS_HEADERS
   mach_egads.hpp
)

target_sources(mach
   PRIVATE
      mach_egads.cpp
      ${MACH_UTILITY_EGADS_HEADERS}
)

# get_target_property(MACH_PUBLIC_HEADERS mach PUBLIC_HEADER)
# set(MACH_PUBLIC_HEADERS ${MACH_PUBLIC_HEADERS} ${MACH_UTILITY_EGADS_HEADERS})
# set_target_properties(mach PROPERTIES
#    PUBLIC_HEADER "${MACH_PUBLIC_HEADERS}"
# )

add_public_headers_to_target(mach "${MACH_UTILITY_EGADS_HEADERS}")

#add_library(mach_egads ${SOURCES})

target_link_libraries(mach PUBLIC ${EGADS_LIBRARIES})

# Include directories
target_include_directories(mach
   PUBLIC
      "${EGADS_INCLUDE_DIR})"
     "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>"
     "$<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDE_DIR}>"
)

# make sure the compiler can find the config header
#target_include_directories(mach PRIVATE
#  $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}>)


# mach_export_library(mach_egads)

endif (MFEM_USE_EGADS)