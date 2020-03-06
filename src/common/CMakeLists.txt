target_sources(mach
   PRIVATE
      common.hpp
      coefficient.cpp
      coefficient.hpp
      default_options.hpp
      evolver.cpp
      evolver.hpp
      inexact_newton.cpp
      inexact_newton.hpp
      orthopoly.cpp
      orthopoly.hpp
      sbp_fe.cpp
      sbp_fe.hpp
      spline.cpp
      spline.hpp
      mesh_movement.cpp
      mesh_movement.hpp
)

target_include_directories(mach
   PUBLIC
      "${CMAKE_CURRENT_SOURCE_DIR}"
)

add_subdirectory(data)
