target_sources(miso
   PRIVATE
      joule.cpp
      joule.hpp
)

target_include_directories(miso
   PUBLIC
      "${CMAKE_CURRENT_SOURCE_DIR}"
)
