#ifndef MISO_DEMAG_FLUX_COEFF
#define MISO_DEMAG_FLUX_COEFF

#include <map>
#include <string>

#include "adept.h"
#include "mfem.hpp"

#include "coefficient.hpp"
#include "miso_input.hpp"

namespace miso
{
class DemagFluxCoefficient : public StateCoefficient
{
public:
   friend void setInputs(DemagFluxCoefficient &current,
                         const MISOInputs &inputs)
   { }

   double Eval(mfem::ElementTransformation &trans,
               const mfem::IntegrationPoint &ip) override;

   double Eval(mfem::ElementTransformation &trans,
               const mfem::IntegrationPoint &ip,
               double state) override;

   double EvalStateDeriv(mfem::ElementTransformation &trans,
                         const mfem::IntegrationPoint &ip,
                         double state) override;

   double EvalState2ndDeriv(mfem::ElementTransformation &trans,
                            const mfem::IntegrationPoint &ip,
                            const double state) override;

   /// TODO: Adapt EvalRevDiff as needed for demag flux
   void EvalRevDiff(const double Q_bar,
                    mfem::ElementTransformation &trans,
                    const mfem::IntegrationPoint &ip,
                    mfem::DenseMatrix &PointMat_bar) override;

   DemagFluxCoefficient(const nlohmann::json &B_knee_options,
                        const nlohmann::json &materials);

private:
   /// The underlying coefficient that does all the heavy lifting
   MeshDependentCoefficient B_knee;
};

}  // namespace miso

#endif