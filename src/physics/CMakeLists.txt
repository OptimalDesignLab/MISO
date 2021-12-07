target_sources(mach
   PRIVATE
      diag_mass_integ.cpp
      diag_mass_integ.hpp
      finite_element_dual.hpp
      finite_element_state.cpp
      finite_element_state.hpp
      finite_element_vector.cpp
      finite_element_vector.hpp
      mach_input.cpp
      mach_input.hpp
      mach_integrator.cpp
      mach_integrator.hpp
      mach_load.hpp
      mach_nonlinearform.cpp
      mach_nonlinearform.hpp
      #mass_operator.cpp
      #mass_operator.hpp
      mfem_common_integ.cpp
      mfem_common_integ.hpp
      pde_solver.cpp
      pde_solver.hpp
      physics.hpp
      solver.cpp
      solver.hpp
)

target_include_directories(mach
   PUBLIC
      "${CMAKE_CURRENT_SOURCE_DIR}"
)

add_subdirectory(electromagnetics)
add_subdirectory(fluidflow)
# add_subdirectory(joule)
add_subdirectory(meshmove)
add_subdirectory(thermal)
