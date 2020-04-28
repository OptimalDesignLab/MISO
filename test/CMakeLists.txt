if(NOT TARGET Catch)
   add_library(Catch INTERFACE)
   target_include_directories(Catch INTERFACE "${CMAKE_SOURCE_DIR}/test") 
   # target_sources(Catch INTERFACE catch.hpp)
endif()

#set(UNIT_TESTS unit)
#add_custom_target("${UNIT_TESTS}")
add_subdirectory(unit)

#set(REGRESSION_TESTS regression)
#add_custom_target("${REGRESSION_TESTS}")
add_subdirectory(regression)

