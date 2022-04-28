set(MACH_PHYSICS_THERMAL_HEADERS
   thermal_integ.hpp
   thermal_residual.hpp
   thermal.hpp
)

target_sources(mach
   PRIVATE
      thermal_integ.cpp
      thermal_residual.cpp
      thermal.cpp
      ${MACH_PHYSICS_THERMAL_HEADERS}
)

add_public_headers_to_target(mach "${MACH_PHYSICS_THERMAL_HEADERS}")

target_include_directories(mach
   PUBLIC
     "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>"
     "$<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDE_DIR}>"
)
