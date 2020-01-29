target_sources(mach
   PRIVATE
      diag_mass_integ.cpp
      diag_mass_integ.hpp
      solver.cpp
      solver.hpp
)

add_subdirectory(electromagnetics)
add_subdirectory(fluidflow)
