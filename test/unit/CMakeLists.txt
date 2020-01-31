function(test_func source_list data_files)
   foreach(X ${source_list})
      add_executable("${X}.bin" "${X}.cpp" unit_test_main.cpp ${data_files})
      target_link_libraries("${X}.bin" mach Catch)
      add_test(NAME "${X}" COMMAND "${X}.bin")
   endforeach()
endfunction(test_func)



#test_func(test_euler_fluxes
#          test_euler_fluxes.bin
#          test_euler_fluxes.cpp)


include_directories(BEFORE ${PROJECT_SOURCE_DIR})
# Include the build directory where mfem.hpp and mfem-performance.hpp are.
#include_directories(BEFORE ${PROJECT_BINARY_DIR})
# Include the source directory for the unit tests - catch.hpp is there.
#include_directories(BEFORE "../${CMAKE_CURRENT_SOURCE_DIR}")

set(UNIT_TESTS_SRCS
   "test_euler_fluxes;test_euler_integ"
)

#foreach(X ${UNIT_TEST_SRCS})
set(UNIT_TEST_SRCS
   test_euler_fluxes 
   test_euler_integ
   test_inexact_newton
   test_navier_stokes_fluxes
   test_navier_stokes_integ
   test_reconstruction
   test_sbp_fe
   test_viscous_integ
)

test_func("${UNIT_TEST_SRCS}" euler_test_data.cpp)

#foreach(X ${UNIT_TEST_SRCS})
#   add_executable("${X}.bin" "${X}.cpp" unit_test_main.cpp euler_test_data.cpp)
#   target_link_libraries("${X}.bin" mach Catch)
#   add_test(NAME "${X}" COMMAND "${X}.bin")
#   message("${X}.bin")
#endforeach()

#add_executable(test_euler_flux unit_test_main.cpp test_euler_fluxes.cpp euler_test_data.cpp)
#target_link_libraries(test_euler_flux
#   PUBLIC
#      mach
#      Catch
#)

#add_test(NAME euler_flux COMMAND test_euler_flux)

#add_executable(test_euler_integ unit_test_main.cpp test_euler_integ.cpp euler_test_data.cpp)
#target_link_libraries(test_euler_integ
#   PUBLIC
#      mach
#      Catch
#      # "${ADEPT_LIBRARIES}" # shouldn't need since mach should take care of the linking
#)

#add_test(NAME euler_integ COMMAND test_euler_integ)


#add_executable(unit_tests ${UNIT_TESTS_SRCS})
#target_link_libraries(unit_tests mach)

#add_dependencies(${UNIT_TESTS} unit_tests)
#add_test(NAME unit_tests COMMAND unit_tests)

