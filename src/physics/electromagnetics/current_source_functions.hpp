#ifndef MACH_CURRENT_SOURCE_FUNCTIONS
#define MACH_CURRENT_SOURCE_FUNCTIONS

#include <memory>

#include "mfem.hpp"
#include "nlohmann/json.hpp"

namespace mach
{
/// Construct vector coefficient that describes the current source direction
/// \param[in] options - JSON options dictionary that maps mesh element
/// attributes to known current source functions
std::unique_ptr<mfem::VectorCoefficient> constructCurrent(
    const nlohmann::json &current_options);

}  // namespace mach

#endif
