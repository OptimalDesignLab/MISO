target_sources(mach
   PRIVATE
      sbp_operators.hpp
      gauss_hermite.hpp
)

target_include_directories(mach
   PUBLIC
      "${CMAKE_CURRENT_SOURCE_DIR}"
)

