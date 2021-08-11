function(create_egads_tests source_list data_files)
   foreach(X ${source_list})
      add_executable("${X}.bin" "${X}.cpp" "../regression_test_main.cpp" ${data_files})
      add_dependencies("${X}.bin" copy-egads-data-files)
      add_dependencies(build_tests "${X}.bin")
      target_link_libraries("${X}.bin" PRIVATE mach "${EGADS_LIBRARIES}" Catch)
      target_include_directories("${X}.bin" PRIVATE "${CMAKE_SOURCE_DIR}" "${EGADS_INCLUDE_DIRS}")
      add_test(NAME "${X}" COMMAND "${X}.bin") 
      # configure_file("${X}_options.json" "${X}_options.json" COPYONLY)
   endforeach()
endfunction(create_egads_tests)

# for all targets created in this file, include the main dir (containing mach.hpp)
#include_directories(BEFORE ${PROJECT_SOURCE_DIR})

# group together all regression tests
set(TEST_SRCS
   test_acloss_functional
   test_meshmovement_egads_cyl
#   test_joule_wire_egads
#   test_team13
#   test_thermal_steady_embedded_box
   test_thermal_ring
#   coupled_motor
   test_coulomb1984_force
   test_torque_loop
#   test_magnet_force
#   test_magnet_box
#   test_femm_tutorial
)

add_custom_target(copy-egads-data-files ALL
                  COMMAND ${CMAKE_COMMAND} -E copy_directory
                          ${CMAKE_CURRENT_SOURCE_DIR}/data
                          ${CMAKE_CURRENT_BINARY_DIR}/data)

create_egads_tests("${TEST_SRCS}" "../regression_data.cpp")
