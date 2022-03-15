set(MACH_UTILITY_HEADERS
   div_free_projector.hpp
   irrotational_projector.hpp
   kdtree.hpp
   mach_types.hpp
   l2_transfer_operator.hpp
   utilities.hpp
   utils.hpp
)

target_sources(mach
   PUBLIC
      div_free_projector.cpp
      irrotational_projector.cpp
      l2_transfer_operator.cpp
      utils.cpp
      ${MACH_UTILITY_HEADERS}
)

# get_target_property(MACH_PUBLIC_HEADERS mach PUBLIC_HEADER)
# set(MACH_PUBLIC_HEADERS ${MACH_PUBLIC_HEADERS} ${MACH_UTILITY_HEADERS})
# set_target_properties(mach PROPERTIES
#    PUBLIC_HEADER "${MACH_PUBLIC_HEADERS}"
# )

add_public_headers_to_target(mach "${MACH_UTILITY_HEADERS}")

target_include_directories(mach
   PUBLIC
     "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>"
     "$<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDE_DIR}>"
)

if (MFEM_USE_EGADS)
   add_subdirectory(EGADS)
endif (MFEM_USE_EGADS)

