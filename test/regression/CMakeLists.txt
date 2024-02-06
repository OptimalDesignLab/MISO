function(create_tests source_list data_files)
   foreach(X ${source_list})
      add_executable("${X}.bin" "${X}.cpp" regression_test_main.cpp ${data_files})
      add_dependencies(build_tests "${X}.bin")
      target_link_libraries("${X}.bin" PRIVATE miso Catch)
      target_include_directories("${X}.bin" PRIVATE "${CMAKE_SOURCE_DIR}")
      add_test(NAME "${X}" COMMAND "${X}.bin") 
      # configure_file("${X}_options.json" "${X}_options.json" COPYONLY)
   endforeach()
endfunction(create_tests)

# for all targets created in this file, include the main dir (containing miso.hpp)
#include_directories(BEFORE ${PROJECT_SOURCE_DIR})

# group together all regression tests
set(REGRESSION_TEST_SRCS
   test_steady_vortex
   test_navier_stokes_mms
   test_navier_stokes3d_mms
   test_inviscid_mms
   test_farfield3d_exact
   test_thermal_cube
   test_steady_thermal_cube
   test_miso_inputs
   test_magnetostatic_box
   test_meshmovement_box
   test_meshmovement_annulus
   test_joule_box
)

create_tests("${REGRESSION_TEST_SRCS}" regression_data.cpp)

if (MFEM_USE_EGADS)
   add_subdirectory(egads)
endif (MFEM_USE_EGADS)
