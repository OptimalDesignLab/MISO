# Creates executables for each file in the SANDBOX_SRCS list
# Accompaning options files must be named the same as the .cpp file,
#   but with _options.json as the end.
# For example, steady_vortex.cpp goes with steady_vortex_options.json
function(create_sandbox source_list)
   foreach(X ${source_list})
      add_executable("${X}.bin" "${X}.cpp")
      add_dependencies(sandbox "${X}.bin")
      target_link_libraries("${X}.bin" PRIVATE mach)
      target_include_directories("${X}.bin" PRIVATE "${CMAKE_SOURCE_DIR}")
      configure_file("${X}_options.json" "${X}_options.json" COPYONLY)
      configure_file("mach_options.json" "mach_options.json" COPYONLY)
   endforeach()
endfunction(create_sandbox)

set(SANDBOX_SRCS
   steady_vortex
   steady_vortex_adjoint
   unsteady_vortex
   viscous_shock
   airfoil_steady
   airfoil_chaotic
   magnetostatic_box
   magnetostatic_motor
   magnetostatic_wire
   navier_stokes_mms
   thermal_square
   surface_distance
   joule_wire
   ns_walltest
   rans_freestream
   rans_walltest
   gf_error
)

create_sandbox("${SANDBOX_SRCS}")
