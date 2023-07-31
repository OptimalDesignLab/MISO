#ifndef MISO_DEFAULT_OPTIONS
#define MISO_DEFAULT_OPTIONS

#include "nlohmann/json.hpp"

namespace miso
{
/// Defines the default options for miso
/// Declared extern and defined in default_options.cpp so that executables can
/// `#include "miso.hpp"` and avoid a duplicate symbol linking error
extern const nlohmann::json default_options;

}  // namespace miso

#endif
