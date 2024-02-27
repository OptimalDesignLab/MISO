set(MISO_MESH_WARP_HEADERS
   mesh_move_integ.hpp
   mesh_warper.hpp
)

target_sources(miso
   PUBLIC
      mesh_move_integ.cpp
      mesh_warper.cpp
      ${MISO_MESH_WARP_HEADERS}
)

add_public_headers_to_target(miso "${MISO_MESH_WARP_HEADERS}")

target_include_directories(miso
   PUBLIC
     "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>"
     "$<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDE_DIR}>"
)
