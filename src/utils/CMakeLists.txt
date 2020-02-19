target_sources(mach
   PRIVATE
      json.hpp
      mach_types.hpp
      utilities.hpp
      utils.cpp
      utils.hpp
)

target_include_directories(mach
   PUBLIC
     "${CMAKE_CURRENT_SOURCE_DIR}"
)

