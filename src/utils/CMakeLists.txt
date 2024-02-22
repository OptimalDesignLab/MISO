set(MISO_UTILITY_HEADERS
   div_free_projector.hpp
   irrotational_projector.hpp
   kdtree.hpp
   miso_types.hpp
   l2_transfer_operator.hpp
   utilities.hpp
   utils.hpp
)

target_sources(miso
   PUBLIC
      div_free_projector.cpp
      irrotational_projector.cpp
      l2_transfer_operator.cpp
      utils.cpp
      ${MISO_UTILITY_HEADERS}
)

add_public_headers_to_target(miso "${MISO_UTILITY_HEADERS}")

target_include_directories(miso
   PUBLIC
     "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>"
     "$<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDE_DIR}>"
)

add_subdirectory(mesh_warper)
