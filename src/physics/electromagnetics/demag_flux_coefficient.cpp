#include <cmath>
#include <memory>
#include <string>
#include <vector>

#include "mfem.hpp"
#include "nlohmann/json.hpp"
#include "tinysplinecxx.h"

#include "demag_flux_coefficient.hpp"
#include "utils.hpp"

#include "coefficient.hpp"
#include "mach_input.hpp"

namespace
{
class LinearTempDepDemagFluxCoefficient : public mach::StateCoefficient
{
public:
   /// \brief Define a demag flux (flux density B at the knee) model that is a
   /// linear function of temperature \param[in] alpha_B_knee - the linear slope
   /// of the flux density at the knee with respect to temperature (T/deg).
   /// Found by first gathering the flux densities of the knees in the second
   /// quadrant, then doing a linear fit to obtain the slope empirically.
   /// \param[in] beta_B_knee - the y-intercept of the linear fit that
   /// approximates the flux density at the knee with respect to temperature
   /// (T). Found by first gathering the flux densities of the knees in the
   /// second quadrant, then doing a linear fit to obtain the y-intercept
   /// empirically.
   LinearTempDepDemagFluxCoefficient(const double &alpha_B_knee,
                                     const double &beta_B_knee);

   /// \brief Evaluate the demag flux in the element described by trans at the
   /// point ip.
   /// \note When this method is called, the caller must make sure that the
   /// IntegrationPoint associated with trans is the same as ip. This can be
   /// achieved by calling trans.SetIntPoint(&ip).
   double Eval(mfem::ElementTransformation &trans,
               const mfem::IntegrationPoint &ip,
               double state) override;

   /// \brief Evaluate the second derivative of the permanent magnetic
   /// demagnetization constraint equation coefficient in the element with
   /// respect to the 1st state variable
   ///   described by trans at the point ip.
   /// \note When this method is called, the caller must make sure that the
   /// IntegrationPoint associated with trans is the same as ip. This can be
   /// achieved by calling trans.SetIntPoint(&ip).
   double EvalStateDeriv(mfem::ElementTransformation &trans,
                         const mfem::IntegrationPoint &ip,
                         double state) override;

   /// \brief Evaluate the second derivative of the demag flux in the element
   /// with respect to the 2nd state variable
   ///   described by trans at the point ip.
   /// \note When this method is called, the caller must make sure that the
   /// IntegrationPoint associated with trans is the same as ip. This can be
   /// achieved by calling trans.SetIntPoint(&ip).
   double EvalState2ndDeriv(mfem::ElementTransformation &trans,
                            const mfem::IntegrationPoint &ip,
                            double state) override;

   /// TODO: Adapt EvalRevDiff if needed for demag flux
   void EvalRevDiff(const double Q_bar,
                    mfem::ElementTransformation &trans,
                    const mfem::IntegrationPoint &ip,
                    mfem::DenseMatrix &PointMat_bar) override
   { }

protected:
   double alpha_B_knee;
   double beta_B_knee;
};

/// TODO: this default demag flux coefficient will need to be altered
/// TODO: this will need some work -- COME BACK TO THIS (possibly even remove)
std::unique_ptr<mfem::Coefficient> constructDefaultDemagFluxCoeff(
    const std::string &material_name,
    const nlohmann::json &materials)
{
   double B_knee = 0.0;
   return std::make_unique<mfem::ConstantCoefficient>(B_knee);
}

// Get Demag parameters (alpha and beta) from JSON
void getDemagParams(const nlohmann::json &material,
                    const nlohmann::json &materials,
                    double &alpha_B_knee,
                    double &beta_B_knee)
{
   const auto &material_name = material["name"].get<std::string>();

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
}

// Construct the demag flux coefficient
std::unique_ptr<mfem::Coefficient> constructDemagFluxCoeff(
    const nlohmann::json &component,
    const nlohmann::json &materials)
{
   std::unique_ptr<mfem::Coefficient>
       temp_coeff;  // temp=temporary, not temperature
   const auto &material = component["material"];

   /// If "material" is a string, it is interpreted to be the name of a
   /// material. We default to a B_knee coeff of 0 ///TODO: (change this value
   /// as needed) unless there is a different value in the material library
   if (material.is_string())
   {
      const auto &material_name = material.get<std::string>();
      temp_coeff = constructDefaultDemagFluxCoeff(material_name, materials);
   }
   else
   {
      const auto &material_name = material["name"].get<std::string>();

      if (material.contains("Demag"))
      {
         // Declare variables
         double alpha_B_knee;
         double beta_B_knee;

         // Obtain the necessary parameters from the JSON
         getDemagParams(material, materials, alpha_B_knee, beta_B_knee);

         // Can now construct the coefficient accordingly
         temp_coeff = std::make_unique<LinearTempDepDemagFluxCoefficient>(
             alpha_B_knee, beta_B_knee);

         /// TODO: Implement this error handing as needed
         // else
         // {
         //    std::string error_msg =
         //          "Insufficient information to compute demag flux coefficient
         //          for material \"";
         //    error_msg += material_name;
         //    error_msg += "\"!\n";
         //    throw mach::MachException(error_msg);
         // }
      }
      else
      {
         // Doesn't have the Demag JSON structure; assign it default coefficient
         temp_coeff = constructDefaultDemagFluxCoeff(material_name, materials);
      }
   }
   return temp_coeff;
}

}  // anonymous namespace

namespace mach
{
double DemagFluxCoefficient::Eval(mfem::ElementTransformation &trans,
                                  const mfem::IntegrationPoint &ip)
{
   return B_knee.Eval(trans, ip);
}

double DemagFluxCoefficient::Eval(mfem::ElementTransformation &trans,
                                  const mfem::IntegrationPoint &ip,
                                  double state)
{
   return B_knee.Eval(trans, ip, state);
}

double DemagFluxCoefficient::EvalStateDeriv(mfem::ElementTransformation &trans,
                                            const mfem::IntegrationPoint &ip,
                                            double state)
{
   return B_knee.EvalStateDeriv(trans, ip, state);
}

double DemagFluxCoefficient::EvalState2ndDeriv(
    mfem::ElementTransformation &trans,
    const mfem::IntegrationPoint &ip,
    const double state)
{
   return B_knee.EvalState2ndDeriv(trans, ip, state);
}

/// TODO: Adapt if keeping, remove if not
void DemagFluxCoefficient::EvalRevDiff(const double Q_bar,
                                       mfem::ElementTransformation &trans,
                                       const mfem::IntegrationPoint &ip,
                                       mfem::DenseMatrix &PointMat_bar)
{
   B_knee.EvalRevDiff(Q_bar, trans, ip, PointMat_bar);
}

/// TODO: Change B_knee(std::make_unique<mfem::ConstantCoefficient>(0.0)) line
/// IF the equivalent lines... std::unique_ptr<mfem::Coefficient>
/// constructDefaultDemagFluxCoeff( from earlier change
DemagFluxCoefficient::DemagFluxCoefficient(const nlohmann::json &B_knee_options,
                                           const nlohmann::json &materials)
 : B_knee(std::make_unique<mfem::ConstantCoefficient>(0.0))
{
   if (B_knee_options.contains("components"))
   {
      /// Options are being passed in. Loop over the components within and
      /// construct a demag flux coefficient for each
      for (const auto &component : B_knee_options["components"])
      {
         int attr = component.value("attr", -1);
         if (-1 != attr)
         {
            B_knee.addCoefficient(
                attr, constructDefaultDemagFluxCoeff(component, materials));
         }
         else
         {
            for (const auto &attribute : component["attrs"])
            {
               B_knee.addCoefficient(
                   attribute, constructDemagFluxCoeff(component, materials));
            }
         }
      }
   }
   else
   {
      /// Components themselves are being passed in. Loop over the components
      /// and construct a demag flux coefficient for each
      auto components = B_knee_options;
      for (const auto &component : components)
      {
         int attr = component.value("attr", -1);
         if (-1 != attr)
         {
            B_knee.addCoefficient(
                attr, constructDefaultDemagFluxCoeff(component, materials));
         }
         else
         {
            for (const auto &attribute : component["attrs"])
            {
               B_knee.addCoefficient(
                   attribute, constructDemagFluxCoeff(component, materials));
            }
         }
      }
   }
}

}  // namespace mach

namespace
{
LinearTempDepDemagFluxCoefficient::LinearTempDepDemagFluxCoefficient(
    const double &alpha_B_knee,
    const double &beta_B_knee)
 : alpha_B_knee(alpha_B_knee), beta_B_knee(beta_B_knee)
{ }

double LinearTempDepDemagFluxCoefficient::Eval(
    mfem::ElementTransformation &trans,
    const mfem::IntegrationPoint &ip,
    const double state)
{
   double T = state;  // assuming the state is the temperature in Kelvin
   // Evaluate the value for the demag flux at the knee B_knee
   // std::cout << "B_knee = alpha_B_knee*T+beta_B_knee = " << alpha_B_knee <<
   // "*" << T << "+" << beta_B_knee << "\n";
   auto B_knee =
       alpha_B_knee * T + beta_B_knee;  // the approximate flux density of the
                                        // knee point at the given temperature
   return B_knee;
}

double LinearTempDepDemagFluxCoefficient::EvalStateDeriv(
    mfem::ElementTransformation &trans,
    const mfem::IntegrationPoint &ip,
    const double state)
{
   // double T=state; // assuming the state is the temperature in Kelvin
   // auto B_knee = alpha_B_knee*T+beta_B_knee; // the approximate flux density
   // of the knee point at the given temperature
   auto dB_kneedT = alpha_B_knee;
   return dB_kneedT;
}

double LinearTempDepDemagFluxCoefficient::EvalState2ndDeriv(
    mfem::ElementTransformation &trans,
    const mfem::IntegrationPoint &ip,
    const double state)
{
   // double T=state; // assuming the state is the temperature in Kelvin
   // auto B_knee = alpha_B_knee*T+beta_B_knee; // the approximate flux density
   // of the knee point at the given temperature auto dB_kneedT = alpha_B_knee;
   auto d2B_kneedT2 = 0.0;
   return d2B_kneedT2;
}

}  // namespace