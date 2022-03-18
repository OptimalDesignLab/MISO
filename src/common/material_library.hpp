#ifndef MACH_MATERIAL_LIBRARY
#define MACH_MATERIAL_LIBRARY

#include "nlohmann/json.hpp"

namespace mach
{
/// Defines the material library for for mach
/// Declared extern and defined in default_options.cpp so that executables can
/// `#include "mach.hpp"` and avoid a duplicate symbol linking error
extern const nlohmann::json material_library;

}  // namespace mach

#endif
