function(create_tests source_list data_files)
   foreach(X ${source_list})
      add_executable("${X}.bin" "${X}.cpp" unit_test_main.cpp ${data_files})
      add_dependencies(tests "${X}.bin")
      target_link_libraries("${X}.bin" PRIVATE mach Catch)
      target_include_directories("${X}.bin" PRIVATE "${CMAKE_SOURCE_DIR}")
      add_test(NAME "${X}" COMMAND "${X}.bin") 
   endforeach()
endfunction(create_tests)

# for all targets created in this file, include the main dir (containing mach.hpp)
#include_directories(BEFORE ${PROJECT_SOURCE_DIR})

# group together all fluids related tests
set(FLUID_TEST_SRCS
   test_euler_fluxes 
   test_euler_integ
   test_euler_assemble
   test_euler_sens_integ
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
   test_res_integ
)

# group EM tests
set(EM_TEST_SRCS
   test_electromag_integ
)

create_tests("${FLUID_TEST_SRCS}" euler_test_data.cpp)
create_tests("${EM_TEST_SRCS}" electromag_test_data.hpp)
