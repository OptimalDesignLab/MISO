#ifndef MACH_DEFAULT_OPTIONS
#define MACH_DEFAULT_OPTIONS

#include "nlohmann/json.hpp"

namespace mach
{
/// Defines the default options for mach
/// Declared extern and defined in default_options.cpp so that executables can
/// `#include "mach.hpp"` and avoid a duplicate symbol linking error
extern const nlohmann::json default_options;

}  // namespace mach

#endif
