target_sources(mach
   PRIVATE
      common.hpp
      default_options.hpp
      evolver.cpp
      evolver.hpp
      inexact_newton.cpp
      inexact_newton.hpp
      orthopoly.cpp
      orthopoly.hpp
      sbp_fe.cpp
      sbp_fe.hpp
)

add_subdirectory(data)
