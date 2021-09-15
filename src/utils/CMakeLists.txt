target_sources(mach
   PRIVATE
      div_free_projector.cpp
      div_free_projector.hpp
      irrotational_projector.cpp
      irrotational_projector.hpp
      kdtree.hpp
      mach_types.hpp
      utilities.hpp
      utils.cpp
      utils.hpp
)

target_include_directories(mach
   PUBLIC
     "${CMAKE_CURRENT_SOURCE_DIR}"
)

if (MFEM_USE_EGADS)
   add_subdirectory(EGADS)
endif (MFEM_USE_EGADS)

