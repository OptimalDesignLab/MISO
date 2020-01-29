target_sources(mach
   PRIVATE
      solver.cpp
      solver.hpp
)

add_subdirectory(electromagnetics)
add_subdirectory(fluidflow)
