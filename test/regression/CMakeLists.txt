function(create_tests source_list data_files)
   foreach(X ${source_list})
      add_executable("${X}.bin" "${X}.cpp" regression_test_main.cpp ${data_files})
      add_dependencies(build_tests "${X}.bin")
      target_link_libraries("${X}.bin" PRIVATE mach Catch)
      target_include_directories("${X}.bin" PRIVATE "${CMAKE_SOURCE_DIR}")
      add_test(NAME "${X}" COMMAND "${X}.bin") 
      # configure_file("${X}_options.json" "${X}_options.json" COPYONLY)
   endforeach()
endfunction(create_tests)

# for all targets created in this file, include the main dir (containing mach.hpp)
#include_directories(BEFORE ${PROJECT_SOURCE_DIR})

# group together all regression tests
set(REGRESSION_TEST_SRCS
   test_steady_vortex
   test_navier_stokes_mms
   test_thermal_cube
   test_steady_thermal_cube
   test_mach_inputs
   test_magnetostatic_box
   test_magnetostatic_box2d
   test_meshmovement_box
   test_meshmovement_annulus
   test_ac_loss
   test_magnetostatic_box_new
   # test_2d_magnet_in_box
   test_magnetostatic_residual
   test_thermal_residual
   test_weak_boundary
   test_thermal_solver
)

create_tests("${REGRESSION_TEST_SRCS}" regression_data.cpp)

if (MFEM_USE_EGADS)
   add_subdirectory(egads)
endif (MFEM_USE_EGADS)
