set(MACH_PHYSICS_HEADERS
   common_outputs.hpp
   diag_mass_integ.hpp
   finite_element_dual.hpp
   finite_element_state.hpp
   finite_element_vector.hpp
   mach_input.hpp
   mach_integrator.hpp
   mach_load.hpp
   mach_nonlinearform.hpp
   mfem_common_integ.hpp
   pde_solver.hpp
   physics.hpp
   solver.hpp
)


target_sources(mach
   PRIVATE
      common_outputs.cpp
      diag_mass_integ.cpp
      finite_element_state.cpp
      finite_element_vector.cpp
      mach_input.cpp
      mach_integrator.cpp
      mach_nonlinearform.cpp
      mfem_common_integ.cpp
      pde_solver.cpp
      solver.cpp
      ${MACH_PHYSICS_HEADERS}
)

# get_target_property(MACH_PUBLIC_HEADERS mach PUBLIC_HEADER)
# set(MACH_PUBLIC_HEADERS ${MACH_PUBLIC_HEADERS} ${MACH_PHYSICS_HEADERS})
# set_target_properties(mach PROPERTIES
#    PUBLIC_HEADER "${MACH_PUBLIC_HEADERS}"
# )

add_public_headers_to_target(mach "${MACH_PHYSICS_HEADERS}")

target_include_directories(mach
   PUBLIC
     "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>"
     "$<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDE_DIR}>"
)

add_subdirectory(electromagnetics)
add_subdirectory(fluidflow)
# add_subdirectory(joule)
add_subdirectory(meshmove)
add_subdirectory(thermal)
