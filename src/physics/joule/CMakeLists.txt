target_sources(mach
   PRIVATE
      joule.cpp
      joule.hpp
)

target_include_directories(mach
   PUBLIC
      "${CMAKE_CURRENT_SOURCE_DIR}"
)
