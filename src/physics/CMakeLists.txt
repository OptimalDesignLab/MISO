target_sources(mach
   PRIVATE
      diag_mass_integ.cpp
      diag_mass_integ.hpp
      physics.hpp
      solver.cpp
      solver.hpp
      res_integ.cpp
      res_integ.hpp
)

target_include_directories(mach
   PUBLIC
      "${CMAKE_CURRENT_SOURCE_DIR}"
)

add_subdirectory(electromagnetics)
add_subdirectory(fluidflow)
add_subdirectory(thermal)
