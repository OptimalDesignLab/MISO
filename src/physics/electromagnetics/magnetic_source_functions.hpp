#ifndef MACH_MAGNETIC_SOURCE_FUNCTIONS
#define MACH_MAGNETIC_SOURCE_FUNCTIONS

#include "mfem.hpp"

namespace mach
{
/// Construct vector coefficient that describes the magnetization source
/// direction \param[in] options - JSON options dictionary that maps mesh
/// element attributes to known magnetization source functions
std::unique_ptr<mfem::VectorCoefficient> constructMagnetization(
    const nlohmann::json &mag_options);

}  // namespace mach

#endif
