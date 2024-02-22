#ifndef MISO_REMNANT_FLUX_COEFFICIENT
#define MISO_REMNANT_FLUX_COEFFICIENT

#include <map>
#include <string>

#include "adept.h"
#include "mfem.hpp"

#include "coefficient.hpp"
#include "miso_input.hpp"

namespace miso
{
class RemnantFluxCoefficient : public StateCoefficient
{
public:
   friend void setInputs(RemnantFluxCoefficient &current,
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

   void EvalRevDiff(const double Q_bar,
                    mfem::ElementTransformation &trans,
                    const mfem::IntegrationPoint &ip,
                    double state,
                    mfem::DenseMatrix &PointMat_bar) override;

   RemnantFluxCoefficient(const nlohmann::json &material);

private:
   /// The underlying coefficient that does all the heavy lifting
   MeshDependentCoefficient B_r;
};

}  // namespace miso

#endif