target_sources(mach
   PRIVATE
      electromag_integ.cpp
      electromag_integ.hpp
      electromagnetics.hpp
      magnetostatic.cpp
      magnetostatic.hpp
)

target_include_directories(mach
   PUBLIC
      "${CMAKE_CURRENT_SOURCE_DIR}"
)

