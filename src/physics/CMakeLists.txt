target_sources(mach
   PRIVATE
      diag_mass_integ.cpp
      diag_mass_integ.hpp
      mach_input.cpp
      mach_input.hpp
      mach_integrator.cpp
      mach_integrator.hpp
      mach_load.hpp
      mfem_common_integ.cpp
      mfem_common_integ.hpp
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
add_subdirectory(joule)
add_subdirectory(meshmove)
add_subdirectory(thermal)
