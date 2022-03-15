set(MACH_PHYSICS_FLUIDS_HEADERS
   advection.hpp
   euler_fluxes.hpp
   euler_integ_def.hpp
   euler_integ.hpp
   euler_sens_integ.hpp
   euler_sens_integ_def.hpp
   euler.hpp
   fluidflow.hpp
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

target_sources(mach
   PRIVATE
      advection.cpp
      euler.cpp
      flow_residual.cpp
      flow_solver.cpp
      galer_diff.cpp
      navier_stokes.cpp
      ${MACH_PHYSICS_FLUIDS_HEADERS}
)

# get_target_property(MACH_PUBLIC_HEADERS mach PUBLIC_HEADER)
# set(MACH_PUBLIC_HEADERS ${MACH_PUBLIC_HEADERS} ${MACH_PHYSICS_FLUIDS_HEADERS})
# set_target_properties(mach PROPERTIES
#    PUBLIC_HEADER "${MACH_PUBLIC_HEADERS}"
# )

add_public_headers_to_target(mach "${MACH_PHYSICS_FLUIDS_HEADERS}")

target_include_directories(mach
   PUBLIC
     "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>"
     "$<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDE_DIR}>"
)

