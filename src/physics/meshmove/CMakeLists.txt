set(MISO_PHYSICS_MESHMOVE_HEADERS
   mesh_move_integ.hpp
   mesh_movement.hpp
)

target_sources(miso
   PRIVATE
      mesh_move_integ.cpp
      mesh_movement.cpp
      ${MISO_PHYSICS_MESHMOVE_HEADERS}
)

add_public_headers_to_target(miso "${MISO_PHYSICS_MESHMOVE_HEADERS}")

target_include_directories(miso
   PUBLIC
     "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>"
     "$<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDE_DIR}>"
)

