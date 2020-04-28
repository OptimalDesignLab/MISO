#ifndef MACH_DEFAULT_OPTIONS
#define MACH_DEFAULT_OPTIONS

#include "json.hpp"

namespace mach
{

/// Defines the default options for mach
///
/// This is placed in its own file because it is likely to grow large.  Also,
/// while it would have been nice to use a raw string here to define the default
/// options, this would not have permitted comments.
///
/// Declared extern and defined in default_options.cpp so that executables can
/// `#include "mach.hpp"` and avoid a duplicate symbol linking error
extern const nlohmann::json default_options;

} // namespace mach

#endif
