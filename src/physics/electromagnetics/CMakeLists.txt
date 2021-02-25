target_sources(mach
   PRIVATE
      current_load.cpp
      current_load.hpp
      electromag_integ.cpp
      electromag_integ.hpp
      electromagnetics.hpp
      magnetic_load.cpp
      magnetic_load.hpp
      magnetostatic.cpp
      magnetostatic.hpp
)

target_include_directories(mach
   PUBLIC
      "${CMAKE_CURRENT_SOURCE_DIR}"
)

