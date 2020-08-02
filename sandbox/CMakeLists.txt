# Creates executables for each file in the SANDBOX_SRCS list
# Accompaning options files must be named the same as the .cpp file,
#   but with _options.json as the end.
# For example, steady_vortex.cpp goes with steady_vortex_options.json
function(create_sandbox source_list)
   foreach(X ${source_list})
      add_executable("${X}.bin" "${X}.cpp")
      add_dependencies(sandbox "${X}.bin")
      target_link_libraries("${X}.bin" mach)
      configure_file("${X}_options.json" "${X}_options.json" COPYONLY)
      configure_file("mach_options.json" "mach_options.json" COPYONLY)
   endforeach()
endfunction(create_sandbox)

set(SANDBOX_SRCS
   steady_vortex
   unsteady_vortex
   viscous_shock
   airfoil_steady
   airfoil_chaotic
	galerkin_difference
   steady_vortex_adjoint
   sod_shock
)

create_sandbox("${SANDBOX_SRCS}")
