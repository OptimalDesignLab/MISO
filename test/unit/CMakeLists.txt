#function(test_func testname binname srcname)
#   add_executable(${binname} ${srcname} "${CMAKE_SOURCE_DIR}/mach.hpp")
#   target_link_libraries(${binname} mach Catch)
#   add_test(NAME ${testname} COMMAND ${binname})
#endfunction(test_func)



#test_func(test_euler_fluxes
#          test_euler_fluxes.bin
#          test_euler_fluxes.cpp)


include_directories(BEFORE ${PROJECT_SOURCE_DIR})
# Include the build directory where mfem.hpp and mfem-performance.hpp are.
include_directories(BEFORE ${PROJECT_BINARY_DIR})
# Include the source directory for the unit tests - catch.hpp is there.
include_directories(BEFORE "../${CMAKE_CURRENT_SOURCE_DIR}")

set(UNIT_TESTS_SRCS
   test_euler_fluxes
   test_euler_integ
)

foreach(TEST IN LISTS UNIT_TEST_SRCS)
   add_executable("${TEST}.bin" "${TEST}.cpp")
   target_link_libraries("{TEST}.bin" mach Catch)
   add_test(NAME ${TEST} COMMAND "${TEST}.bin")
   message("hello")
endforeach()

add_executable(test_euler_flux unit_test_main.cpp test_euler_fluxes.cpp euler_test_data.cpp)
target_link_libraries(test_euler_flux
   PUBLIC
      mach
      Catch
)

add_test(NAME euler_flux COMMAND test_euler_flux)

#add_executable(unit_tests ${UNIT_TESTS_SRCS})
#target_link_libraries(unit_tests mach)

#add_dependencies(${UNIT_TESTS} unit_tests)
#add_test(NAME unit_tests COMMAND unit_tests)

