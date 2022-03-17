set(MACH_MESH_WARP_HEADERS
   mesh_warper.hpp
)

target_sources(mach
   PUBLIC
      mesh_warper.cpp
      ${MACH_MESH_WARP_HEADERS}
)

add_public_headers_to_target(mach "${MACH_MESH_WARP_HEADERS}")

target_include_directories(mach
   PUBLIC
     "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>"
     "$<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDE_DIR}>"
)
