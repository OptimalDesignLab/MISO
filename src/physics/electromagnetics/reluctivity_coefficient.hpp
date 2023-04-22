#ifndef MACH_RELUCTIVITY_COEFFICIENT
#define MACH_RELUCTIVITY_COEFFICIENT

#include <map>
#include <string>

#include "adept.h"
#include "mfem.hpp"

#include "coefficient.hpp"
#include "mach_input.hpp"

namespace mach
{
class ReluctivityCoefficient : public StateCoefficient
{
public:
   friend void setInputs(ReluctivityCoefficient &current,
                         const MachInputs &inputs)
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

   ReluctivityCoefficient(const nlohmann::json &nu_options,
                          const nlohmann::json &materials);

private:
   /// The underlying coefficient that does all the heavy lifting
   MeshDependentCoefficient nu;
};

}  // namespace mach

#endif
