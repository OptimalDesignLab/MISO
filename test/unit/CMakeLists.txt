function(test_func testname binname srcname)
   add_executable(${binname} ${srcname})
   target_link_libraries(${binname} mach Catch)
   add_test(NAME ${testname} COMMAND ${binname})
endfunction(test_func)



test_func(test_euler_fluxes
          test_euler_fluxes.bin
          test_euler_fluxes.cpp)
