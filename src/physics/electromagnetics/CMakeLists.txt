set(MACH_PHYSICS_ELECTROMAGNETICS_HEADERS
   current_load.hpp
   current_source_functions.hpp
   electromag_integ.hpp
   electromag_outputs.hpp
   electromagnetics.hpp
   magnetic_load.hpp
   magnetic_source_functions.hpp
   magnetostatic_load.hpp
   magnetostatic_residual.hpp
   magnetostatic.hpp
   reluctivity_coefficient.hpp
   conductivity_coefficient.hpp
   cal2_ke_coefficient.hpp
   cal2_kh_coefficient.hpp
   pm_demag_constraint_coeff.hpp
)

target_sources(mach
   PRIVATE
      current_load.cpp
      current_source_functions.cpp
      electromag_integ.cpp
      electromag_outputs.cpp
      magnetic_load.cpp
      magnetic_source_functions.cpp
      magnetostatic_residual.cpp
      magnetostatic.cpp
      reluctivity_coefficient.cpp
      conductivity_coefficient.cpp
      cal2_ke_coefficient.cpp
      cal2_kh_coefficient.cpp
      pm_demag_constraint_coeff.cpp
      ${MACH_PHYSICS_ELECTROMAGNETICS_HEADERS}
)

# get_target_property(MACH_PUBLIC_HEADERS mach PUBLIC_HEADER)
# set(MACH_PUBLIC_HEADERS ${MACH_PUBLIC_HEADERS} ${MACH_PHYSICS_ELECTROMAGNETICS_HEADERS})
# set_target_properties(mach PROPERTIES
#    PUBLIC_HEADER "${MACH_PUBLIC_HEADERS}"
# )

add_public_headers_to_target(mach "${MACH_PHYSICS_ELECTROMAGNETICS_HEADERS}")

target_include_directories(mach
   PUBLIC
     "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>"
     "$<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDE_DIR}>"
)

