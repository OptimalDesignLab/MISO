target_sources(mach
   PRIVATE
      thermal.cpp
      thermal.hpp
      therm_integ.cpp
      therm_integ.hpp
      temp_integ.cpp
      temp_integ.hpp
      thermal_verify.cpp
)

target_include_directories(mach
   PUBLIC
      "${CMAKE_CURRENT_SOURCE_DIR}"
)
