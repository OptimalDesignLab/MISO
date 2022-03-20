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
      centgridfunc.cpp
      centgridfunc.hpp
      rbfgridfunc.cpp
      rbfgridfunc.hpp
      bfgsnewton.hpp
      bfgsnewton.cpp
)

target_include_directories(mach
   PUBLIC
      "${CMAKE_CURRENT_SOURCE_DIR}"
)

add_subdirectory(data)
