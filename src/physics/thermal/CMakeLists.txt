set(MISO_PHYSICS_THERMAL_HEADERS
   temp_integ.hpp
   thermal_integ.hpp
   thermal_residual.hpp
   thermal.hpp
)

target_sources(miso
   PRIVATE
      temp_integ.hpp
      thermal_integ.cpp
      thermal_residual.cpp
      thermal.cpp
      ${MISO_PHYSICS_THERMAL_HEADERS}
)

add_public_headers_to_target(miso "${MISO_PHYSICS_THERMAL_HEADERS}")

target_include_directories(miso
   PUBLIC
     "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>"
     "$<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDE_DIR}>"
)
