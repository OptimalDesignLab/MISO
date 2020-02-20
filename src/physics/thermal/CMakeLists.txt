target_sources(mach
   PRIVATE
      thermal.cpp
      thermal.hpp
      therm_integ.cpp
      therm_integ.hpp
)

target_include_directories(mach
   PUBLIC
      "${CMAKE_CURRENT_SOURCE_DIR}"
)
