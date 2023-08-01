if (MFEM_USE_EGADS)

option(EGADS_LITE "Enable EGADSlite" OFF)
# this file brings EGADS_LITE from CMake to C++
# configure_file("${CMAKE_CURRENT_SOURCE_DIR}/miso_egads_config.hpp.in"
#               "${CMAKE_CURRENT_BINARY_DIR}/miso_egads_config.hpp")

target_sources(miso
   PRIVATE
      miso_egads.cpp
      miso_egads.hpp
)

#add_library(miso_egads ${SOURCES})

target_link_libraries(miso PUBLIC ${EGADS_LIBRARIES})

# Include directories
target_include_directories(miso 
   PUBLIC
      "${CMAKE_CURRENT_SOURCE_DIR}"
      "${EGADS_INCLUDE_DIR})"
)

# make sure the compiler can find the config header
#target_include_directories(miso PRIVATE
#  $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}>)


# miso_export_library(miso_egads)

endif (MFEM_USE_EGADS)