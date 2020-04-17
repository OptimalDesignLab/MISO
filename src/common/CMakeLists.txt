target_sources(mach
   PRIVATE
      coefficient.cpp
      coefficient.hpp
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
      pfem_extras.cpp
      pfem_extras.hpp
      sbp_fe.cpp
      sbp_fe.hpp
      spline.cpp
      spline.hpp
)

target_include_directories(mach
   PUBLIC
      "${CMAKE_CURRENT_SOURCE_DIR}"
)

add_subdirectory(data)
