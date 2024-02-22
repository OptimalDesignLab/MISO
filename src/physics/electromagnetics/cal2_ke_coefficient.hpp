#ifndef MISO_CAL2_KE_COEFFICIENT
#define MISO_CAL2_KE_COEFFICIENT

#include <map>
#include <string>

#include "adept.h"
#include "mfem.hpp"

#include "coefficient.hpp"
#include "miso_input.hpp"

namespace miso
{
class CAL2keCoefficient : public ThreeStateCoefficient
{
public:
   friend void setInputs(CAL2keCoefficient &current, const MISOInputs &inputs)
   { }

   double Eval(mfem::ElementTransformation &trans,
               const mfem::IntegrationPoint &ip) override;

   double Eval(mfem::ElementTransformation &trans,
               const mfem::IntegrationPoint &ip,
               double state1,
               double state2,
               double state3) override;

   double EvalDerivS1(mfem::ElementTransformation &trans,
                      const mfem::IntegrationPoint &ip,
                      double state1,
                      double state2,
                      double state3) override;

   double EvalDerivS2(mfem::ElementTransformation &trans,
                      const mfem::IntegrationPoint &ip,
                      double state1,
                      double state2,
                      double state3) override;

   double EvalDerivS3(mfem::ElementTransformation &trans,
                      const mfem::IntegrationPoint &ip,
                      double state1,
                      double state2,
                      double state3) override;

   double Eval2ndDerivS1(mfem::ElementTransformation &trans,
                         const mfem::IntegrationPoint &ip,
                         double state1,
                         double state2,
                         double state3) override;

   double Eval2ndDerivS2(mfem::ElementTransformation &trans,
                         const mfem::IntegrationPoint &ip,
                         double state1,
                         double state2,
                         double state3) override;

   double Eval2ndDerivS3(mfem::ElementTransformation &trans,
                         const mfem::IntegrationPoint &ip,
                         double state1,
                         double state2,
                         double state3) override;

   double Eval2ndDerivS1S2(mfem::ElementTransformation &trans,
                           const mfem::IntegrationPoint &ip,
                           double state1,
                           double state2,
                           double state3) override;

   double Eval2ndDerivS1S3(mfem::ElementTransformation &trans,
                           const mfem::IntegrationPoint &ip,
                           double state1,
                           double state2,
                           double state3) override;

   double Eval2ndDerivS2S3(mfem::ElementTransformation &trans,
                           const mfem::IntegrationPoint &ip,
                           double state1,
                           double state2,
                           double state3) override;

   /// TODO: Likely not necessary because of Eval2ndDerivS1S2
   double Eval2ndDerivS2S1(mfem::ElementTransformation &trans,
                           const mfem::IntegrationPoint &ip,
                           double state1,
                           double state2,
                           double state3) override;

   /// TODO: Likely not necessary because of Eval2ndDerivS1S3
   double Eval2ndDerivS3S1(mfem::ElementTransformation &trans,
                           const mfem::IntegrationPoint &ip,
                           double state1,
                           double state2,
                           double state3) override;

   /// TODO: Likely not necessary because of Eval2ndDerivS2S3
   double Eval2ndDerivS3S2(mfem::ElementTransformation &trans,
                           const mfem::IntegrationPoint &ip,
                           double state1,
                           double state2,
                           double state3) override;

   /// TODO: Adapt EvalRevDiff as needed for Core Losses
   void EvalRevDiff(const double Q_bar,
                    mfem::ElementTransformation &trans,
                    const mfem::IntegrationPoint &ip,
                    mfem::DenseMatrix &PointMat_bar) override;

   CAL2keCoefficient(const nlohmann::json &coreloss_options,
                     const nlohmann::json &materials);

private:
   /// The underlying coefficient that does all the heavy lifting
   MeshDependentThreeStateCoefficient CAL2_ke;
};

}  // namespace miso

#endif