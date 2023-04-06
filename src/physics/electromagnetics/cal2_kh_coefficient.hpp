#ifndef MACH_CAL2_KH_COEFFICIENT
#define MACH_CAL2_KH_COEFFICIENT

#include <map>
#include <string>

#include "adept.h"
#include "mfem.hpp"

#include "coefficient.hpp"
#include "mach_input.hpp"

namespace mach
{
class CAL2khCoefficient : public ThreeStateCoefficient
{
public:
   friend void setInputs(CAL2khCoefficient &current, const MachInputs &inputs)
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

   CAL2khCoefficient(const nlohmann::json &coreloss_options,
                     const nlohmann::json &materials);

private:
   /// The underlying coefficient that does all the heavy lifting
   MeshDependentThreeStateCoefficient CAL2_kh;
};

}  // namespace mach

#endif