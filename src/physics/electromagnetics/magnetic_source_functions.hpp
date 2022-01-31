#ifndef MACH_MAGNETIC_SOURCE_FUNCTIONS
#define MACH_MAGNETIC_SOURCE_FUNCTIONS

#include <map>
#include <string>

#include "adept.h"
#include "mfem.hpp"

#include "coefficient.hpp"
#include "mach_input.hpp"

namespace mach
{
class MagnetizationCoefficient : public mfem::VectorCoefficient
{
public:
   friend void setInputs(MagnetizationCoefficient &current,
                         const MachInputs &inputs)
   { }

   void Eval(mfem::Vector &V,
             mfem::ElementTransformation &trans,
             const mfem::IntegrationPoint &ip) override;

   void EvalRevDiff(const mfem::Vector &V_bar,
                    mfem::ElementTransformation &trans,
                    const mfem::IntegrationPoint &ip,
                    mfem::DenseMatrix &PointMat_bar) override;

   MagnetizationCoefficient(adept::Stack &diff_stack,
                            const nlohmann::json &magnet_options,
                            const nlohmann::json &materials,
                            int vdim = 3);

private:
   /// The underlying coefficient that does all the heavy lifting
   VectorMeshDependentCoefficient mag_coeff;
   /// Map that holds the remnant flux for each magnet material group
   std::map<std::string, double> remnant_flux_map;
};

}  // namespace mach

#endif
