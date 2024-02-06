function(create_tests source_list data_files)
   foreach(X ${source_list})
      add_executable("${X}.bin" "${X}.cpp" unit_test_main.cpp ${data_files})
      add_dependencies(build_tests "${X}.bin")
      target_link_libraries("${X}.bin" PRIVATE miso Catch)
      target_include_directories("${X}.bin" PRIVATE "${CMAKE_SOURCE_DIR}")
      add_test(NAME "${X}" COMMAND "${X}.bin") 
   endforeach()
endfunction(create_tests)

function(create_mpi_tests source_list data_files)
   foreach(X ${source_list})
      add_executable("${X}.bin" "${X}.cpp" unit_test_mpi_main.cpp ${data_files})
      add_dependencies(build_tests "${X}.bin")
      target_link_libraries("${X}.bin" PRIVATE miso Catch)
      target_include_directories("${X}.bin" PRIVATE "${CMAKE_SOURCE_DIR}")
      add_test(NAME "${X}" COMMAND "${X}.bin") 
   endforeach()
endfunction(create_mpi_tests)

# for all targets created in this file, include the main dir (containing miso.hpp)
#include_directories(BEFORE ${PROJECT_SOURCE_DIR})

# group together all fluids related tests
set(FLUID_TEST_SRCS
   test_euler_fluxes 
   test_euler_integ
   test_euler_assemble
   test_euler_sens_integ
   test_element_integrator
   test_evolver
   test_inexact_newton
   test_navier_stokes_fluxes
   test_navier_stokes_integ
   test_navier_stokes_assemble
   test_reconstruction
   test_sbp_fe
   test_viscous_integ
   test_surface
   test_utils
   test_thermal_integ
)

# group EM tests
set(EM_TEST_SRCS
)

# group fluid MPI tests
set(FLUID_MPI_TEST_SRCS
   test_flow_control_residual
   test_flow_residual
   test_3dflow_residual
   test_flow_solver
   test_mfem_common_integ
   test_miso_residual
)

# group EM MPI tests
set(EM_MPI_TEST_SRCS
   test_electromag_integ
   test_irrotational_projector
   test_div_free_projector
   test_magnetostatic_solver
   test_steinmetz_integ
   test_coefficient
   test_miso_load
   test_current_load
   test_magnetic_load
   test_magnetostatic_load
   #test_mass_operator
   test_common_outputs
   test_abstract_solver
   test_pde_solver
   test_data_logging
   test_l2_transfer_operator
   test_mesh_warper
   test_miso_nonlinearform
)

create_tests("${FLUID_TEST_SRCS}" euler_test_data.cpp)
create_tests("${EM_TEST_SRCS}" electromag_test_data.hpp)

create_mpi_tests("${EM_MPI_TEST_SRCS}" electromag_test_data.hpp)
create_mpi_tests("${FLUID_MPI_TEST_SRCS}" euler_test_data.cpp)
