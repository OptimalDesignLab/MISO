function(create_tests source_list data_files)
   foreach(X ${source_list})
      add_executable("${X}.bin" "${X}.cpp" regression_test_main.cpp ${data_files})
      add_dependencies(tests "${X}.bin")
      target_link_libraries("${X}.bin" PRIVATE mach Catch)
      target_include_directories("${X}.bin" PRIVATE "${CMAKE_SOURCE_DIR}")
      add_test(NAME "${X}" COMMAND "${X}.bin") 
      configure_file("${X}_options.json" "${X}_options.json" COPYONLY)
   endforeach()
endfunction(create_tests)

# for all targets created in this file, include the main dir (containing mach.hpp)
#include_directories(BEFORE ${PROJECT_SOURCE_DIR})

# group together all fluids related tests
set(FLUID_TEST_SRCS
    test_steady_vortex
)

# group together all thermal related tests
set(THERMAL_TEST_SRCS
    test_thermal_cube
)


create_tests("${FLUID_TEST_SRCS}" regression_data.cpp)
create_tests("${THERMAL_TEST_SRCS}" regression_data.cpp)
