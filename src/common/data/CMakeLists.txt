target_sources(miso
   PRIVATE
      sbp_operators.hpp
)

target_include_directories(miso
   PUBLIC
      "${CMAKE_CURRENT_SOURCE_DIR}"
)

