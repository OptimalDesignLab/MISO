target_sources(mach
   PRIVATE
      poisson_dg_def_cut.hpp
      poisson_dg_cut.hpp
     )

target_include_directories(mach
   PUBLIC
      "${CMAKE_CURRENT_SOURCE_DIR}"
)

