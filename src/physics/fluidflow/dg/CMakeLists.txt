target_sources(mach
   PRIVATE
      euler_integ_def_dg.hpp
      euler_integ_dg.hpp
      euler_dg.cpp
      euler_dg.hpp 
      inviscid_integ_def_dg.hpp
      inviscid_integ_dg.hpp
      mms_integ_def_dg.hpp
      mms_integ_dg.hpp
     )

target_include_directories(mach
   PUBLIC
      "${CMAKE_CURRENT_SOURCE_DIR}"
)

