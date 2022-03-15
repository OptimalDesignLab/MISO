set(MACH_COMMON_DATA_HEADERS
   sbp_operators.hpp
)

target_sources(mach
   PRIVATE
   ${MACH_COMMON_DATA_HEADERS}
)

add_public_headers_to_target(mach "${MACH_COMMON_DATA_HEADERS}")

target_include_directories(mach
   PUBLIC
     "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>"
     "$<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDE_DIR}>"
)
