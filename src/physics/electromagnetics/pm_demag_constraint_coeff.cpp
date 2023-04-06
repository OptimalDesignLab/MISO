#include <cmath>
#include <memory>
#include <string>
#include <vector>

#include "mfem.hpp"
#include "nlohmann/json.hpp"
#include "tinysplinecxx.h"

#include "pm_demag_constraint_coeff.hpp"
#include "utils.hpp"

#include "coefficient.hpp"
#include "mach_input.hpp"

namespace
{
class PMDemagConstraintEqCoeff : public mach::TwoStateCoefficient
{
public:
   /// \brief Define a model to represent the permanent magnetic demagnetization
   /// constraint equation coefficient \param[in] T0 - the reference temperature
   /// (K). Given by the manufacturer. \param[in] alpha_B_r - the remanent flux
   /// temperature coefficient (%/deg). Given by the manufacturer. \param[in]
   /// B_r_T0 - the remanent flux at the given reference temperature (T). Given
   /// by the manufacturer. \param[in] alpha_H_ci - the intrinsic coercivity
   /// temperature coefficient (%/deg). Given by the manufacturer. \param[in]
   /// H_ci_T0 - the intrinsic coercivity at the given reference temperature
   /// (kA/m). Given by the manufacturer. \param[in] alpha_B_knee - the linear
   /// slope of the flux density at the knee with respect to temperature
   /// (T/deg). Found by first gathering the flux densities of the knees in the
   /// second quadrant, then doing a linear fit to obtain the slope empirically.
   /// \param[in] beta_B_knee - the y-intercept of the linear fit that
   /// approximates the flux density at the knee with respect to temperature
   /// (T). Found by first gathering the flux densities of the knees in the
   /// second quadrant, then doing a linear fit to obtain the y-intercept
   /// empirically. \param[in] alpha_H_knee - the linear slope of the coercive
   /// force at the knee with respect to temperature (kA/m per deg). Found by
   /// first gathering the coercive forces of the knees in the second quadrant,
   /// then doing a linear fit to obtain the slope empirically. \param[in]
   /// beta_H_knee - the y-intercept of the linear fit that approximates the
   /// coercive forces at the knee with respect to temperature (kA/m). Found by
   /// first gathering the coercive forces of the knees in the second quadrant,
   /// then doing a linear fit to obtain the y-intercept empirically.
   PMDemagConstraintEqCoeff(const double &T0,
                            const double &alpha_B_r,
                            const double &B_r_T0,
                            const double &alpha_H_ci,
                            const double &H_ci_T0,
                            const double &alpha_B_knee,
                            const double &beta_B_knee,
                            const double &alpha_H_knee,
                            const double &beta_H_knee);

   /// \brief Evaluate the permanent magnetic demagnetization constraint
   /// equation coefficient in the element described by trans at the point ip.
   /// \note When this method is called, the caller must make sure that the
   /// IntegrationPoint associated with trans is the same as ip. This can be
   /// achieved by calling trans.SetIntPoint(&ip).
   double Eval(mfem::ElementTransformation &trans,
               const mfem::IntegrationPoint &ip,
               double state1,
               double state2) override;

   /// \brief Evaluate the derivative of the permanent magnetic demagnetization
   /// constraint equation coefficient in the element with respect to the 1st
   /// state variable
   ///   described by trans at the point ip.
   /// \note When this method is called, the caller must make sure that the
   /// IntegrationPoint associated with trans is the same as ip. This can be
   /// achieved by calling trans.SetIntPoint(&ip).
   double EvalDerivS1(mfem::ElementTransformation &trans,
                      const mfem::IntegrationPoint &ip,
                      double state1,
                      double state2) override;

   /// \brief Evaluate the derivative of the permanent magnetic demagnetization
   /// constraint equation coefficient in the element with respect to the 2nd
   /// state variable
   ///   described by trans at the point ip.
   /// \note When this method is called, the caller must make sure that the
   /// IntegrationPoint associated with trans is the same as ip. This can be
   /// achieved by calling trans.SetIntPoint(&ip).
   double EvalDerivS2(mfem::ElementTransformation &trans,
                      const mfem::IntegrationPoint &ip,
                      double state1,
                      double state2) override;

   /// \brief Evaluate the second derivative of the permanent magnetic
   /// demagnetization constraint equation coefficient in the element with
   /// respect to the 1st state variable
   ///   described by trans at the point ip.
   /// \note When this method is called, the caller must make sure that the
   /// IntegrationPoint associated with trans is the same as ip. This can be
   /// achieved by calling trans.SetIntPoint(&ip).
   double Eval2ndDerivS1(mfem::ElementTransformation &trans,
                         const mfem::IntegrationPoint &ip,
                         double state1,
                         double state2) override;

   /// \brief Evaluate the derivative of the permanent magnetic demagnetization
   /// constraint equation coefficient in the element with respect to the 2nd
   /// state variable
   ///   described by trans at the point ip.
   /// \note When this method is called, the caller must make sure that the
   /// IntegrationPoint associated with trans is the same as ip. This can be
   /// achieved by calling trans.SetIntPoint(&ip).
   double Eval2ndDerivS2(mfem::ElementTransformation &trans,
                         const mfem::IntegrationPoint &ip,
                         double state1,
                         double state2) override;

   /// \brief Evaluate the second derivative of the permanent magnetic
   /// demagnetization constraint equation coefficient in the element with
   /// respect to the 1st then 2nd state variable
   ///   described by trans at the point ip.
   /// \note When this method is called, the caller must make sure that the
   /// IntegrationPoint associated with trans is the same as ip. This can be
   /// achieved by calling trans.SetIntPoint(&ip).
   double Eval2ndDerivS1S2(mfem::ElementTransformation &trans,
                           const mfem::IntegrationPoint &ip,
                           double state1,
                           double state2) override;

   /// TODO: Likely not necessary because of Eval2ndDerivS1S2
   ///  \brief Evaluate the derivative of the permanent magnetic demagnetization
   ///  constraint equation coefficient in the element with respect to the 2nd
   ///  then 1st state variable
   ///    described by trans at the point ip.
   ///  \note When this method is called, the caller must make sure that the
   ///  IntegrationPoint associated with trans is the same as ip. This can be
   ///  achieved by calling trans.SetIntPoint(&ip).
   double Eval2ndDerivS2S1(mfem::ElementTransformation &trans,
                           const mfem::IntegrationPoint &ip,
                           double state1,
                           double state2) override;

   /// TODO: Adapt EvalRevDiff if needed for permanent magnetic demagnetization
   /// constraint equation coefficient
   void EvalRevDiff(const double Q_bar,
                    mfem::ElementTransformation &trans,
                    const mfem::IntegrationPoint &ip,
                    mfem::DenseMatrix &PointMat_bar) override
   { }

protected:
   double T0;
   double alpha_B_r;
   double B_r_T0;
   double alpha_H_ci;
   double H_ci_T0;
   double alpha_B_knee;
   double beta_B_knee;
   double alpha_H_knee;
   double beta_H_knee;
};

/// TODO: this default permanent magnetic demagnetization constraint coefficient
/// will need to be altered
/// TODO: this will need some work -- COME BACK TO THIS (possibly even remove)
std::unique_ptr<mfem::Coefficient> constructDefaultPMDemagConstraintCoeff(
    const std::string &material_name,
    const nlohmann::json &materials)
{
   double PMDemagConstraint = 0.0;
   return std::make_unique<mfem::ConstantCoefficient>(PMDemagConstraint);
}

// Get Demag parameters (T0, alphas, betas, and values at reference
// temperatures) from JSON
void getDemagParams(const nlohmann::json &material,
                    const nlohmann::json &materials,
                    double &T0,
                    double &alpha_B_r,
                    double &B_r_T0,
                    double &alpha_H_ci,
                    double &H_ci_T0,
                    double &alpha_B_knee,
                    double &beta_B_knee,
                    double &alpha_H_knee,
                    double &beta_H_knee)
{
   const auto &material_name = material["name"].get<std::string>();

   // Assign T0 based on material options, else refer to material library
   if (material["Demag"].contains("T0"))
   {
      T0 = material["Demag"]["T0"].get<double>();
   }
   else
   {
      T0 = materials[material_name]["Demag"]["T0"].get<double>();
   }

   // Assign alpha_B_r based on material options, else refer to material library
   if (material["Demag"].contains("alpha_B_r"))
   {
      alpha_B_r = material["Demag"]["alpha_B_r"].get<double>();
   }
   else
   {
      alpha_B_r = materials[material_name]["Demag"]["alpha_B_r"].get<double>();
   }

   // Assign B_r_T0 based on material options, else refer to material library
   if (material["Demag"].contains("B_r_T0"))
   {
      B_r_T0 = material["Demag"]["B_r_T0"].get<double>();
   }
   else
   {
      B_r_T0 = materials[material_name]["Demag"]["B_r_T0"].get<double>();
   }

   // Assign alpha_H_ci based on material options, else refer to material
   // library
   if (material["Demag"].contains("alpha_H_ci"))
   {
      alpha_H_ci = material["Demag"]["alpha_H_ci"].get<double>();
   }
   else
   {
      alpha_H_ci =
          materials[material_name]["Demag"]["alpha_H_ci"].get<double>();
   }

   // Assign H_ci_T0 based on material options, else refer to material library
   if (material["Demag"].contains("H_ci_T0"))
   {
      H_ci_T0 = material["Demag"]["H_ci_T0"].get<double>();
   }
   else
   {
      H_ci_T0 = materials[material_name]["Demag"]["H_ci_T0"].get<double>();
   }

   // Assign alpha_B_knee based on material options, else refer to material
   // library
   if (material["Demag"].contains("alpha_B_knee"))
   {
      alpha_B_knee = material["Demag"]["alpha_B_knee"].get<double>();
   }
   else
   {
      alpha_B_knee =
          materials[material_name]["Demag"]["alpha_B_knee"].get<double>();
   }

   // Assign beta_B_knee based on material options, else refer to material
   // library
   if (material["Demag"].contains("beta_B_knee"))
   {
      beta_B_knee = material["Demag"]["beta_B_knee"].get<double>();
   }
   else
   {
      beta_B_knee =
          materials[material_name]["Demag"]["beta_B_knee"].get<double>();
   }

   // Assign alpha_H_knee based on material options, else refer to material
   // library
   if (material["Demag"].contains("alpha_H_knee"))
   {
      alpha_H_knee = material["Demag"]["alpha_H_knee"].get<double>();
   }
   else
   {
      alpha_H_knee =
          materials[material_name]["Demag"]["alpha_H_knee"].get<double>();
   }

   // Assign beta_H_knee based on material options, else refer to material
   // library
   if (material["Demag"].contains("beta_H_knee"))
   {
      beta_H_knee = material["Demag"]["beta_H_knee"].get<double>();
   }
   else
   {
      beta_H_knee =
          materials[material_name]["Demag"]["beta_H_knee"].get<double>();
   }
}

// Construct the permanent magnetic demagnetization constraint coefficient
std::unique_ptr<mfem::Coefficient> constructPMDemagConstraintCoeff(
    const nlohmann::json &component,
    const nlohmann::json &materials)
{
   std::unique_ptr<mfem::Coefficient>
       temp_coeff;  // temp=temporary, not temperature
   const auto &material = component["material"];

   /// If "material" is a string, it is interpreted to be the name of a
   /// material. We default to a PMDemagConstraint coeff of 0 ///TODO: (change
   /// this value as needed) unless there is a different value in the material
   /// library
   if (material.is_string())
   {
      const auto &material_name = material.get<std::string>();
      temp_coeff =
          constructDefaultPMDemagConstraintCoeff(material_name, materials);
   }
   else
   {
      const auto &material_name = material["name"].get<std::string>();

      if (material.contains("Demag"))
      {
         // Declare variables
         double T0;
         double alpha_B_r;
         double B_r_T0;
         double alpha_H_ci;
         double H_ci_T0;
         double alpha_B_knee;
         double beta_B_knee;
         double alpha_H_knee;
         double beta_H_knee;

         // Obtain the necessary parameters from the JSON
         getDemagParams(material,
                        materials,
                        T0,
                        alpha_B_r,
                        B_r_T0,
                        alpha_H_ci,
                        H_ci_T0,
                        alpha_B_knee,
                        beta_B_knee,
                        alpha_H_knee,
                        beta_H_knee);

         // Can now construct the coefficient accordingly
         temp_coeff = std::make_unique<PMDemagConstraintEqCoeff>(T0,
                                                                 alpha_B_r,
                                                                 B_r_T0,
                                                                 alpha_H_ci,
                                                                 H_ci_T0,
                                                                 alpha_B_knee,
                                                                 beta_B_knee,
                                                                 alpha_H_knee,
                                                                 beta_H_knee);

         /// TODO: Implement this error handing as needed
         // else
         // {
         //    std::string error_msg =
         //          "Insufficient information to compute permanent magnetic
         //          demagnetization constraint coefficient for material \"";
         //    error_msg += material_name;
         //    error_msg += "\"!\n";
         //    throw mach::MachException(error_msg);
         // }
      }
      else
      {
         // Doesn't have the Demag JSON structure; assign it default coefficient
         temp_coeff =
             constructDefaultPMDemagConstraintCoeff(material_name, materials);
      }
   }
   return temp_coeff;
}

}  // anonymous namespace

namespace mach
{
double PMDemagConstraintCoefficient::Eval(mfem::ElementTransformation &trans,
                                          const mfem::IntegrationPoint &ip)
{
   return PMDemagConstraint.Eval(trans, ip);
}

double PMDemagConstraintCoefficient::Eval(mfem::ElementTransformation &trans,
                                          const mfem::IntegrationPoint &ip,
                                          double state1,
                                          double state2)
{
   return PMDemagConstraint.Eval(trans, ip, state1, state2);
}

double PMDemagConstraintCoefficient::EvalDerivS1(
    mfem::ElementTransformation &trans,
    const mfem::IntegrationPoint &ip,
    double state1,
    double state2)
{
   return PMDemagConstraint.EvalDerivS1(trans, ip, state1, state2);
}

double PMDemagConstraintCoefficient::EvalDerivS2(
    mfem::ElementTransformation &trans,
    const mfem::IntegrationPoint &ip,
    double state1,
    double state2)
{
   return PMDemagConstraint.EvalDerivS2(trans, ip, state1, state2);
}

double PMDemagConstraintCoefficient::Eval2ndDerivS1(
    mfem::ElementTransformation &trans,
    const mfem::IntegrationPoint &ip,
    double state1,
    double state2)
{
   return PMDemagConstraint.Eval2ndDerivS1(trans, ip, state1, state2);
}

double PMDemagConstraintCoefficient::Eval2ndDerivS2(
    mfem::ElementTransformation &trans,
    const mfem::IntegrationPoint &ip,
    double state1,
    double state2)
{
   return PMDemagConstraint.Eval2ndDerivS2(trans, ip, state1, state2);
}

double PMDemagConstraintCoefficient::Eval2ndDerivS1S2(
    mfem::ElementTransformation &trans,
    const mfem::IntegrationPoint &ip,
    double state1,
    double state2)
{
   return PMDemagConstraint.Eval2ndDerivS1S2(trans, ip, state1, state2);
}

/// TODO: Likely not necessary because of Eval2ndDerivS1S2
double PMDemagConstraintCoefficient::Eval2ndDerivS2S1(
    mfem::ElementTransformation &trans,
    const mfem::IntegrationPoint &ip,
    double state1,
    double state2)
{
   return PMDemagConstraint.Eval2ndDerivS2S1(trans, ip, state1, state2);
}

/// TODO: Adapt if needed
void PMDemagConstraintCoefficient::EvalRevDiff(
    const double Q_bar,
    mfem::ElementTransformation &trans,
    const mfem::IntegrationPoint &ip,
    mfem::DenseMatrix &PointMat_bar)
{
   PMDemagConstraint.EvalRevDiff(Q_bar, trans, ip, PointMat_bar);
}

/// TODO: Change
/// PMDemagConstraint(std::make_unique<mfem::ConstantCoefficient>(0.0)) line IF
/// the equivalent lines... std::unique_ptr<mfem::Coefficient>
/// constructDefaultPMDemagConstraintCoeff( from earlier change
PMDemagConstraintCoefficient::PMDemagConstraintCoefficient(
    const nlohmann::json &PMDemagConstraint_options,
    const nlohmann::json &materials)
 : PMDemagConstraint(std::make_unique<mfem::ConstantCoefficient>(0.0))
{
   if (PMDemagConstraint_options.contains("components"))
   {
      /// Options are being passed in. Loop over the components within and
      /// construct a PMDemagConstraint coefficient for each
      for (const auto &component : PMDemagConstraint_options["components"])
      {
         int attr = component.value("attr", -1);
         if (-1 != attr)
         {
            PMDemagConstraint.addCoefficient(
                attr,
                constructDefaultPMDemagConstraintCoeff(component, materials));
         }
         else
         {
            for (const auto &attribute : component["attrs"])
            {
               PMDemagConstraint.addCoefficient(
                   attribute,
                   constructPMDemagConstraintCoeff(component, materials));
            }
         }
      }
   }
   else
   {
      /// Components themselves are being passed in. Loop over the components
      /// and construct a PMDemagConstraint coefficient for each
      auto components = PMDemagConstraint_options;
      for (const auto &component : components)
      {
         int attr = component.value("attr", -1);
         if (-1 != attr)
         {
            PMDemagConstraint.addCoefficient(
                attr,
                constructDefaultPMDemagConstraintCoeff(component, materials));
         }
         else
         {
            for (const auto &attribute : component["attrs"])
            {
               PMDemagConstraint.addCoefficient(
                   attribute,
                   constructPMDemagConstraintCoeff(component, materials));
            }
         }
      }
   }
}

}  // namespace mach

namespace
{
PMDemagConstraintEqCoeff::PMDemagConstraintEqCoeff(const double &T0,
                                                   const double &alpha_B_r,
                                                   const double &B_r_T0,
                                                   const double &alpha_H_ci,
                                                   const double &H_ci_T0,
                                                   const double &alpha_B_knee,
                                                   const double &beta_B_knee,
                                                   const double &alpha_H_knee,
                                                   const double &beta_H_knee)
 : T0(T0),
   alpha_B_r(alpha_B_r),
   B_r_T0(B_r_T0),
   alpha_H_ci(alpha_H_ci),
   H_ci_T0(H_ci_T0),
   alpha_B_knee(alpha_B_knee),
   beta_B_knee(beta_B_knee),
   alpha_H_knee(alpha_H_knee),
   beta_H_knee(beta_H_knee)
{
   // No calculations needed for protected class members
}

double PMDemagConstraintEqCoeff::Eval(mfem::ElementTransformation &trans,
                                      const mfem::IntegrationPoint &ip,
                                      const double state1,
                                      const double state2)
{
   // Assuming state1=flux density and state2=temperature
   auto B = state1;
   auto T = state2;

   // Compute the values of the subequations needed for the overall constraint
   // equation
   auto B_knee =
       alpha_B_knee * T + beta_B_knee;  // the approximate flux density of the
                                        // knee point at the given temperature
   auto H_knee = alpha_H_knee * T +
                 beta_H_knee;  // the approximate magnetic field intensity of
                               // the knee point at the given temperature
   auto B_r =
       B_r_T0 * (1 + (alpha_B_r / 100) * (T - T0));  // the remnant flux density
   auto H_ci = H_ci_T0 * (1 + (alpha_H_ci / 100) *
                                  (T - T0));  // the intrinisic coercivity

   // Compute the value of the constraint equation
   // If positive -> reversible demagnetization region.
   // If negative -> irreversible demagnetization region.
   auto C_BT = ((H_knee / (B_knee - B_r)) - ((H_knee - H_ci) / B_knee)) * B -
               (H_ci + ((B_r * H_knee) / (B_knee - B_r)));
   return C_BT;
}

double PMDemagConstraintEqCoeff::EvalDerivS1(mfem::ElementTransformation &trans,
                                             const mfem::IntegrationPoint &ip,
                                             const double state1,
                                             const double state2)
{
   // Assuming state1=flux density and state2=temperature
   // First derivative with respect to flux density

   // auto B = state1;
   auto T = state2;

   // Compute the values of the subequations needed for the overall constraint
   // equation
   auto B_knee =
       alpha_B_knee * T + beta_B_knee;  // the approximate flux density of the
                                        // knee point at the given temperature
   auto H_knee = alpha_H_knee * T +
                 beta_H_knee;  // the approximate magnetic field intensity of
                               // the knee point at the given temperature
   auto B_r =
       B_r_T0 * (1 + (alpha_B_r / 100) * (T - T0));  // the remnant flux density
   auto H_ci = H_ci_T0 * (1 + (alpha_H_ci / 100) *
                                  (T - T0));  // the intrinisic coercivity

   // Compute the value of the constraint equation
   // If positive -> reversible demagnetization region.
   // If negative -> irreversible demagnetization region.
   // auto C_BT = ((H_knee/(B_knee-B_r))-((H_knee-H_ci)/B_knee))*B
   // -(H_ci+((B_r*H_knee)/(B_knee-B_r)));

   auto dC_BTdB = ((H_knee / (B_knee - B_r)) - ((H_knee - H_ci) / B_knee));
   return dC_BTdB;
}

double PMDemagConstraintEqCoeff::EvalDerivS2(mfem::ElementTransformation &trans,
                                             const mfem::IntegrationPoint &ip,
                                             const double state1,
                                             const double state2)
{
   // Assuming state1=flux density and state2=temperature
   // First derivative with respect to temperature

   auto B = state1;
   auto T = state2;

   // Compute the values of the subequations needed for the overall constraint
   // equation
   auto B_knee =
       alpha_B_knee * T + beta_B_knee;  // the approximate flux density of the
                                        // knee point at the given temperature
   auto H_knee = alpha_H_knee * T +
                 beta_H_knee;  // the approximate magnetic field intensity of
                               // the knee point at the given temperature
   auto B_r =
       B_r_T0 * (1 + (alpha_B_r / 100) * (T - T0));  // the remnant flux density
   auto H_ci = H_ci_T0 * (1 + (alpha_H_ci / 100) *
                                  (T - T0));  // the intrinisic coercivity
   // auto C_BT = ((H_knee/(B_knee-B_r))-((H_knee-H_ci)/B_knee))*B
   // -(H_ci+((B_r*H_knee)/(B_knee-B_r)));

   // Used MATLAB to symbolically solve for derivative; will verify analytical
   // derivative in test using FD
   auto SubEq1 = alpha_B_knee - (B_r_T0 * alpha_B_r) / 100;
   auto SubEq2 = B_knee - B_r;
   auto SubEq3 = B_r / B_r_T0;
   auto dC_BTdT =
       B * ((alpha_H_knee / SubEq2) -
            ((alpha_H_knee - (H_ci_T0 * alpha_H_ci / 100)) / B_knee) -
            ((SubEq1 * H_knee) / std::pow(SubEq2, 2)) +
            ((alpha_B_knee * (H_knee - H_ci)) / std::pow(B_knee, 2))) -
       (H_ci_T0 * alpha_H_ci) / 100 -
       ((B_r_T0 * alpha_B_r * H_knee) / (100 * SubEq2)) -
       (B_r_T0 * alpha_H_knee * SubEq3) / SubEq2 +
       (B_r_T0 * SubEq3 * SubEq1 * H_knee) / std::pow(SubEq2, 2);
   return dC_BTdT;
}

double PMDemagConstraintEqCoeff::Eval2ndDerivS1(
    mfem::ElementTransformation &trans,
    const mfem::IntegrationPoint &ip,
    const double state1,
    const double state2)
{
   // Assuming state1=flux density and state2=temperature
   // Second derivative with respect to flux density

   // auto B = state1;
   // auto T = state2;

   // auto C_BT = ((H_knee/(B_knee-B_r))-((H_knee-H_ci)/B_knee))*B
   // -(H_ci+((B_r*H_knee)/(B_knee-B_r)));
   // auto dC_BTdB = ((H_knee/(B_knee-B_r))-((H_knee-H_ci)/B_knee));
   auto d2C_BTdB2 =
       0;  // dC_BTdB is no longer dependent on B, so derivative w/r/t B is 0
   return d2C_BTdB2;
}

double PMDemagConstraintEqCoeff::Eval2ndDerivS2(
    mfem::ElementTransformation &trans,
    const mfem::IntegrationPoint &ip,
    const double state1,
    const double state2)
{
   // Assuming state1=flux density and state2=temperature
   // Second derivative with respect to temperature

   auto B = state1;
   auto T = state2;

   // Compute the values of the subequations needed for the overall constraint
   // equation
   auto B_knee =
       alpha_B_knee * T + beta_B_knee;  // the approximate flux density of the
                                        // knee point at the given temperature
   auto H_knee = alpha_H_knee * T +
                 beta_H_knee;  // the approximate magnetic field intensity of
                               // the knee point at the given temperature
   auto B_r =
       B_r_T0 * (1 + (alpha_B_r / 100) * (T - T0));  // the remnant flux density
   auto H_ci = H_ci_T0 * (1 + (alpha_H_ci / 100) *
                                  (T - T0));  // the intrinisic coercivity
   // auto C_BT = ((H_knee/(B_knee-B_r))-((H_knee-H_ci)/B_knee))*B
   // -(H_ci+((B_r*H_knee)/(B_knee-B_r)));

   // Used MATLAB to symbolically solve for derivative; will verify analytical
   // derivative in test using FD
   auto SubEq1 = alpha_B_knee - (B_r_T0 * alpha_B_r) / 100;
   auto SubEq2 = B_knee - B_r;
   auto SubEq3 = B_r / B_r_T0;
   // auto dC_BTdT = B*(
   // (alpha_H_knee/SubEq2)-((alpha_H_knee-(H_ci_T0*alpha_H_ci/100))/B_knee)
   //                   -((SubEq1*H_knee)/std::pow(SubEq2,2))
   //                   +((alpha_B_knee*(H_knee-H_ci))/std::pow(B_knee,2))   )
   //                -(H_ci_T0*alpha_H_ci)/100
   //                -((B_r_T0*alpha_B_r*H_knee)/(100*SubEq2))
   //                -(B_r_T0*alpha_H_knee*SubEq3)/SubEq2 +
   //                (B_r_T0*SubEq3*SubEq1*H_knee)/std::pow(SubEq2,2);

   // Used MATLAB to symbolically solve for derivative; will verify analytical
   // derivative in test using FD
   auto d2C_BTdT2 =
       (B_r_T0 * alpha_H_knee * SubEq3 * SubEq1 * 2) / std::pow(SubEq2, 2) -
       ((B_r_T0 * alpha_B_r * alpha_H_knee) / (50 * SubEq2)) -
       (B_r_T0 * SubEq3 * std::pow(SubEq1, 2) * H_knee * 2) /
           std::pow(SubEq2, 3) -
       B * ((std::pow(alpha_B_knee, 2) * (H_knee - H_ci) * 2) /
                std::pow(B_knee, 3) -
            (2 * std::pow(SubEq1, 2) * H_knee) / std::pow(SubEq2, 3) -
            (alpha_B_knee * (alpha_H_knee - (H_ci_T0 * alpha_H_ci) / 100) * 2) /
                std::pow(B_knee, 2) +
            (alpha_H_knee * SubEq1 * 2) / std::pow(SubEq2, 2)) +
       ((B_r_T0 * alpha_B_r * SubEq1 * H_knee) / (50 * std::pow(SubEq2, 2)));
   return d2C_BTdT2;
}

double PMDemagConstraintEqCoeff::Eval2ndDerivS1S2(
    mfem::ElementTransformation &trans,
    const mfem::IntegrationPoint &ip,
    const double state1,
    const double state2)
{
   // Assuming state1=flux density and state2=temperature
   // Derivative with respect to flux density then temperature

   // auto B = state1;
   auto T = state2;

   // Compute the values of the subequations needed for the overall constraint
   // equation
   auto B_knee =
       alpha_B_knee * T + beta_B_knee;  // the approximate flux density of the
                                        // knee point at the given temperature
   auto H_knee = alpha_H_knee * T +
                 beta_H_knee;  // the approximate magnetic field intensity of
                               // the knee point at the given temperature
   auto B_r =
       B_r_T0 * (1 + (alpha_B_r / 100) * (T - T0));  // the remnant flux density
   auto H_ci = H_ci_T0 * (1 + (alpha_H_ci / 100) *
                                  (T - T0));  // the intrinisic coercivity

   // Compute the value of the constraint equation
   // If positive -> reversible demagnetization region.
   // If negative -> irreversible demagnetization region.
   // auto C_BT = ((H_knee/(B_knee-B_r))-((H_knee-H_ci)/B_knee))*B
   // -(H_ci+((B_r*H_knee)/(B_knee-B_r)));
   // auto dC_BTdB = ((H_knee/(B_knee-B_r))-((H_knee-H_ci)/B_knee));

   // Used MATLAB to symbolically solve for derivative; will verify analytical
   // derivative in test using FD
   auto SubEq1 = alpha_B_knee - (B_r_T0 * alpha_B_r) / 100;
   auto SubEq2 = B_knee - B_r;
   // auto SubEq3 = B_r/B_r_T0;
   // auto dC_BTdT = B*(
   // (alpha_H_knee/SubEq2)-((alpha_H_knee-(H_ci_T0*alpha_H_ci/100))/B_knee)
   //                   -((SubEq1*H_knee)/std::pow(SubEq2,2))
   //                   +((alpha_B_knee*(H_knee-H_ci))/std::pow(B_knee,2))   )
   //                -(H_ci_T0*alpha_H_ci)/100
   //                -((B_r_T0*alpha_B_r*H_knee)/(100*SubEq2))
   //                -(B_r_T0*alpha_H_knee*SubEq3)/SubEq2 +
   //                (B_r_T0*SubEq3*SubEq1*H_knee)/std::pow(SubEq2,2);

   // Used MATLAB to symbolically solve for derivative; will verify analytical
   // derivative in test using FD
   auto d2C_BTdBdT = (alpha_H_knee / SubEq2) -
                     ((alpha_H_knee - (H_ci_T0 * alpha_H_ci / 100)) / B_knee) -
                     ((SubEq1 * H_knee) / std::pow(SubEq2, 2)) +
                     ((alpha_B_knee * (H_knee - H_ci)) / std::pow(B_knee, 2));
   return d2C_BTdBdT;
}

/// TODO: Likely not necessary because of Eval2ndDerivS2S1
double PMDemagConstraintEqCoeff::Eval2ndDerivS2S1(
    mfem::ElementTransformation &trans,
    const mfem::IntegrationPoint &ip,
    const double state1,
    const double state2)
{
   // Assuming state1=flux density and state2=temperature
   // Derivative with respect to temperature then flux density

   // auto B = state1;
   auto T = state2;

   // Compute the values of the subequations needed for the overall constraint
   // equation
   auto B_knee =
       alpha_B_knee * T + beta_B_knee;  // the approximate flux density of the
                                        // knee point at the given temperature
   auto H_knee = alpha_H_knee * T +
                 beta_H_knee;  // the approximate magnetic field intensity of
                               // the knee point at the given temperature
   auto B_r =
       B_r_T0 * (1 + (alpha_B_r / 100) * (T - T0));  // the remnant flux density
   auto H_ci = H_ci_T0 * (1 + (alpha_H_ci / 100) *
                                  (T - T0));  // the intrinisic coercivity

   // Compute the value of the constraint equation
   // If positive -> reversible demagnetization region.
   // If negative -> irreversible demagnetization region.
   // auto C_BT = ((H_knee/(B_knee-B_r))-((H_knee-H_ci)/B_knee))*B
   // -(H_ci+((B_r*H_knee)/(B_knee-B_r)));
   // auto dC_BTdB = ((H_knee/(B_knee-B_r))-((H_knee-H_ci)/B_knee));

   // Used MATLAB to symbolically solve for derivative; will verify analytical
   // derivative in test using FD
   auto SubEq1 = alpha_B_knee - (B_r_T0 * alpha_B_r) / 100;
   auto SubEq2 = B_knee - B_r;
   // auto SubEq3 = B_r/B_r_T0;
   // auto dC_BTdT = B*(
   // (alpha_H_knee/SubEq2)-((alpha_H_knee-(H_ci_T0*alpha_H_ci/100))/B_knee)
   //                   -((SubEq1*H_knee)/std::pow(SubEq2,2))
   //                   +((alpha_B_knee*(H_knee-H_ci))/std::pow(B_knee,2))   )
   //                -(H_ci_T0*alpha_H_ci)/100
   //                -((B_r_T0*alpha_B_r*H_knee)/(100*SubEq2))
   //                -(B_r_T0*alpha_H_knee*SubEq3)/SubEq2 +
   //                (B_r_T0*SubEq3*SubEq1*H_knee)/std::pow(SubEq2,2);

   // Used MATLAB to symbolically solve for derivative; will verify analytical
   // derivative in test using FD
   auto d2C_BTdTdB = (alpha_H_knee / SubEq2) -
                     ((alpha_H_knee - (H_ci_T0 * alpha_H_ci / 100)) / B_knee) -
                     ((SubEq1 * H_knee) / std::pow(SubEq2, 2)) +
                     ((alpha_B_knee * (H_knee - H_ci)) / std::pow(B_knee, 2));
   return d2C_BTdTdB;
}

/// TODO: is there a need to code EvalRevDiff for PM demag constraint equation
/// coefficient here? I'm thinking not

}  // namespace