#ifndef MISO_EGADS
#define MISO_EGADS

#include <string>

namespace mfem
{
class HypreParVector;
}  // namespace mfem

namespace miso
{
void mapSurfaceMesh(const std::string &old_model_file,
                    const std::string &new_model_file,
                    const std::string &tess_file,
                    mfem::HypreParVector &displacement);
}  // namespace miso

#endif
