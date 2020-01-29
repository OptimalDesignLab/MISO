if(NOT TARGET Catch)
   add_library(Catch INTERFACE)
   # add_dependencies(Catch Catch-External)
   target_include_directories(Catch INTERFACE "${CMAKE_SOURCE_DIR}/test") 
   # target_sources(Catch INTERFACE catch.hpp)
endif()

add_subdirectory(unit)
add_subdirectory(regression)
