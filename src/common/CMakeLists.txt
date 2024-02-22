set(MISO_COMMON_HEADERS
   abstract_solver.hpp
   coefficient.hpp
   common.hpp
   data_logging.hpp
   default_options.hpp
   evolver.hpp
   functional_output.hpp
   inexact_newton.hpp
   linesearch.hpp
   miso_linearform.hpp
   miso_output.hpp
   miso_residual.hpp
   material_library.hpp
   matrix_operators.hpp
   mfem_extensions.hpp
   ode.hpp
   orthopoly.hpp
   relaxed_newton.hpp
   sbp_fe.hpp
   surface.hpp
   surface_def.hpp
)

target_sources(miso
   PRIVATE
      abstract_solver.cpp
      coefficient.cpp
      default_options.cpp
      evolver.cpp
      functional_output.cpp
      inexact_newton.cpp
      linesearch.cpp
      miso_linearform.cpp
      material_library.cpp
      matrix_operators.cpp
      mfem_extensions.cpp
      ode.cpp
      orthopoly.cpp
      relaxed_newton.cpp
      sbp_fe.cpp
      ${MISO_COMMON_HEADERS}
)

add_public_headers_to_target(miso "${MISO_COMMON_HEADERS}")

target_include_directories(miso
   PUBLIC
     "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>"
     "$<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDE_DIR}>"
)

add_subdirectory(data)
