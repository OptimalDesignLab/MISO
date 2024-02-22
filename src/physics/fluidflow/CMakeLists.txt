set(MISO_PHYSICS_FLUIDS_HEADERS
   advection.hpp
   euler_fluxes.hpp
   euler_integ_def.hpp
   euler_integ.hpp
   euler_sens_integ.hpp
   euler_sens_integ_def.hpp
   euler.hpp
   fluidflow.hpp
   flow_control_residual.hpp
   flow_control_solver.hpp
   flow_residual.hpp
   flow_solver.hpp
   galer_diff.hpp
   inviscid_integ_def.hpp
   inviscid_integ.hpp
   mesh_sens_integ.hpp
   mesh_sens_integ_def.hpp
   mms_integ.hpp
   mms_integ_def.hpp
   navier_stokes_fluxes.hpp
   navier_stokes_integ_def.hpp
   navier_stokes_integ.hpp
   navier_stokes.hpp
   viscous_integ_def.hpp
   viscous_integ.hpp
)

target_sources(miso
   PRIVATE
      advection.cpp
      euler.cpp
      flow_control_residual.cpp 
      flow_control_solver.cpp 
      flow_residual.cpp
      flow_solver.cpp
      galer_diff.cpp
      navier_stokes.cpp
      ${MISO_PHYSICS_FLUIDS_HEADERS}
)

add_public_headers_to_target(miso "${MISO_PHYSICS_FLUIDS_HEADERS}")

target_include_directories(miso
   PUBLIC
     "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>"
     "$<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDE_DIR}>"
)

