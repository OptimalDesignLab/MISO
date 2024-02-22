set(MISO_PHYSICS_ELECTROMAGNETICS_HEADERS
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
   remnant_flux_coefficient.hpp
   demag_flux_coefficient.hpp
)

target_sources(miso
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
      remnant_flux_coefficient.cpp
      demag_flux_coefficient.cpp
      ${MISO_PHYSICS_ELECTROMAGNETICS_HEADERS}
)

add_public_headers_to_target(miso "${MISO_PHYSICS_ELECTROMAGNETICS_HEADERS}")

target_include_directories(miso
   PUBLIC
     "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>"
     "$<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDE_DIR}>"
)

