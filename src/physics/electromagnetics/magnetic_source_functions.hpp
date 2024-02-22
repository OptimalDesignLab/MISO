#ifndef MISO_MAGNETIC_SOURCE_FUNCTIONS
#define MISO_MAGNETIC_SOURCE_FUNCTIONS

#include <map>
#include <string>

#include "adept.h"
#include "mfem.hpp"

#include "coefficient.hpp"
#include "miso_input.hpp"

#include "remnant_flux_coefficient.hpp"

namespace miso
{
class MagnetizationCoefficient : public VectorStateCoefficient
{
public:
   friend void setInputs(MagnetizationCoefficient &mag_coeff,
                         const MISOInputs &inputs)
   { }

   void Eval(mfem::Vector &V,
             mfem::ElementTransformation &trans,
             const mfem::IntegrationPoint &ip) override;

   void Eval(mfem::Vector &V,
             mfem::ElementTransformation &trans,
             const mfem::IntegrationPoint &ip,
             double state) override;

   void EvalStateDeriv(mfem::Vector &vec_dot,
                       mfem::ElementTransformation &trans,
                       const mfem::IntegrationPoint &ip,
                       double state) override;

   void EvalRevDiff(const mfem::Vector &V_bar,
                    mfem::ElementTransformation &trans,
                    const mfem::IntegrationPoint &ip,
                    double state,
                    mfem::DenseMatrix &PointMat_bar) override;

   MagnetizationCoefficient(adept::Stack &diff_stack,
                            const nlohmann::json &magnet_options,
                            const nlohmann::json &materials,
                            int vdim = 3);

private:
   /// The underlying coefficient that does all the heavy lifting
   VectorMeshDependentStateCoefficient mag_coeff;
   /// Map that holds the remnant flux for each magnet material group
   std::map<std::string, RemnantFluxCoefficient> remnant_flux_coeffs;
   /// Map that owns all of the underlying magnetization direction coefficients
   std::map<int, mfem::VectorFunctionCoefficient> mag_direction_coeffs;
};

}  // namespace miso

#endif
