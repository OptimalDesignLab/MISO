#ifndef MISO_MATERIAL_LIBRARY
#define MISO_MATERIAL_LIBRARY

#include "nlohmann/json.hpp"

namespace miso
{
/// Defines the material library for for miso
/// Declared extern and defined in default_options.cpp so that executables can
/// `#include "miso.hpp"` and avoid a duplicate symbol linking error
extern const nlohmann::json material_library;

}  // namespace miso

#endif
