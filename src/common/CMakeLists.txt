set(MACH_COMMON_HEADERS
   abstract_solver.hpp
   coefficient.hpp
   common.hpp
   data_logging.hpp
   default_options.hpp
   evolver.hpp
   functional_output.hpp
   inexact_newton.hpp
   mach_linearform.hpp
   mach_output.hpp
   mach_residual.hpp
   material_library.hpp
   matrix_operators.hpp
   mfem_extensions.hpp
   ode.hpp
   orthopoly.hpp
   sbp_fe.hpp
   surface.hpp
   surface_def.hpp
)

target_sources(mach
   PRIVATE
      abstract_solver.cpp
      coefficient.cpp
      default_options.cpp
      evolver.cpp
      functional_output.cpp
      inexact_newton.cpp
      mach_linearform.cpp
      material_library.cpp
      matrix_operators.cpp
      mfem_extensions.cpp
      ode.cpp
      orthopoly.cpp
      sbp_fe.cpp
      ${MACH_COMMON_HEADERS}
)

add_public_headers_to_target(mach "${MACH_COMMON_HEADERS}")

target_include_directories(mach
   PUBLIC
     "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>"
     "$<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDE_DIR}>"
)

add_subdirectory(data)
