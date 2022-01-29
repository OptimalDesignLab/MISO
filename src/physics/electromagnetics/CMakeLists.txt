target_sources(mach
   PRIVATE
      current_load.cpp
      current_load.hpp
      current_source_functions.cpp
      current_source_functions.hpp
      electromag_integ.cpp
      electromag_integ.hpp
      electromag_outputs.hpp
      electromagnetics.hpp
      magnetic_load.cpp
      magnetic_load.hpp
      magnetic_source_functions.cpp
      magnetic_source_functions.hpp
      magnetostatic_load.hpp
      magnetostatic_residual.hpp
      magnetostatic.cpp
      magnetostatic.hpp
)

target_include_directories(mach
   PUBLIC
      "${CMAKE_CURRENT_SOURCE_DIR}"
)

