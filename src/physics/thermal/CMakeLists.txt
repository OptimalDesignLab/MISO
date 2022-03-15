set(MACH_PHYSICS_THERMAL_HEADERS
   thermal.hpp
   therm_integ.hpp
   temp_integ.hpp
)

target_sources(mach
   PRIVATE
      thermal.cpp
      therm_integ.cpp
      temp_integ.cpp
      thermal_verify.cpp
      ${MACH_PHYSICS_THERMAL_HEADERS}
)

# get_target_property(MACH_PUBLIC_HEADERS mach PUBLIC_HEADER)
# set(MACH_PUBLIC_HEADERS ${MACH_PUBLIC_HEADERS} ${MACH_PHYSICS_THERMAL_HEADERS})
# set_target_properties(mach PROPERTIES
#    PUBLIC_HEADER "${MACH_PUBLIC_HEADERS}"
# )

add_public_headers_to_target(mach "${MACH_PHYSICS_THERMAL_HEADERS}")

target_include_directories(mach
   PUBLIC
     "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>"
     "$<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDE_DIR}>"
)
