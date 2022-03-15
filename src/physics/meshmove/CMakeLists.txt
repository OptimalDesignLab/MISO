set(MACH_PHYSICS_MESHMOVE_HEADERS
   mesh_move_integ.hpp
   mesh_movement.hpp
)

target_sources(mach
   PRIVATE
      mesh_move_integ.cpp
      mesh_movement.cpp
      ${MACH_PHYSICS_MESHMOVE_HEADERS}
)

# get_target_property(MACH_PUBLIC_HEADERS mach PUBLIC_HEADER)
# set(MACH_PUBLIC_HEADERS ${MACH_PUBLIC_HEADERS} ${MACH_PHYSICS_MESHMOVE_HEADERS})
# set_target_properties(mach PROPERTIES
#    PUBLIC_HEADER "${MACH_PUBLIC_HEADERS}"
# )

add_public_headers_to_target(mach "${MACH_PHYSICS_MESHMOVE_HEADERS}")

target_include_directories(mach
   PUBLIC
     "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>"
     "$<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDE_DIR}>"
)

