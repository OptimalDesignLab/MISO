target_sources(mach
   PRIVATE
      mesh_move_integ.cpp
      mesh_move_integ.hpp
      mesh_movement.cpp
      mesh_movement.hpp
)

target_include_directories(mach
   PUBLIC
      "${CMAKE_CURRENT_SOURCE_DIR}"
)

