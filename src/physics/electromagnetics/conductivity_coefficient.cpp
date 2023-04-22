#include <cmath>
#include <memory>
#include <string>
#include <vector>

#include "mfem.hpp"
#include "nlohmann/json.hpp"
#include "tinysplinecxx.h"

#include "conductivity_coefficient.hpp"
#include "utils.hpp"

namespace
{
class LinearTempDepConductivityCoefficient : public mach::StateCoefficient
{
public:
   /// \brief Define a conductivity model that is a linear function of
   /// temperature \param[in] alpha_resistivity - temperature dependent
   /// resistivity coefficient \param[in] T_ref - reference temperature
   /// \param[in] sigma_T_ref - the conductivity at the reference temperature
   LinearTempDepConductivityCoefficient(const double &alpha_resistivity,
                                        const double &T_ref,
                                        const double &sigma_T_ref);

   /// \brief Evaluate the conductivity in the element described by trans at the
   /// point ip.
   /// \note When this method is called, the caller must make sure that the
   /// IntegrationPoint associated with trans is the same as ip. This can be
   /// achieved by calling trans.SetIntPoint(&ip).
   double Eval(mfem::ElementTransformation &trans,
               const mfem::IntegrationPoint &ip,
               double state) override;

   /// \brief Evaluate the derivative of conductivity with respsect to
   /// Temperature (T) in the element described by trans at the point ip. \note
   /// When this method is called, the caller must make sure that the
   /// IntegrationPoint associated with trans is the same as ip. This can be
   /// achieved by calling trans.SetIntPoint(&ip).
   double EvalStateDeriv(mfem::ElementTransformation &trans,
                         const mfem::IntegrationPoint &ip,
                         double state) override;

   /// \brief Evaluate the conductivity in the element described by trans at the
   /// point ip.
   /// \note When this method is called, the caller must make sure that the
   /// IntegrationPoint associated with trans is the same as ip. This can be
   /// achieved by calling trans.SetIntPoint(&ip).
   /// TODO: Determine if this is necessary to keep
   double EvalState2ndDeriv(mfem::ElementTransformation &trans,
                            const mfem::IntegrationPoint &ip,
                            const double state) override;

   void EvalRevDiff(const double Q_bar,
                    mfem::ElementTransformation &trans,
                    const mfem::IntegrationPoint &ip,
                    double state,
                    mfem::DenseMatrix &PointMat_bar) override
   { }

protected:
   // alpha_resistivity, T_ref, and sigma_T_ref are needed for methods
   double alpha_resistivity, T_ref, sigma_T_ref;
};

/// Serves as a default value for the conductivity
std::unique_ptr<mfem::Coefficient> constructConstantConductivityCoeff(
    const std::string &material_name,
    const nlohmann::json &materials)
{
   /// TODO: Make sure agrees with material library json structure at the end of
   /// the day
   /// TODO: Decide if want to change default value of sigma to something other
   /// than what was there originally
   auto sigma = materials[material_name].value("sigma", 58.14e6);
   return std::make_unique<mfem::ConstantCoefficient>(sigma);
}

// Function to extract the values for alpha, T_ref, and sigma from the material
// library JSON structure
void getAlphaAndT_RefAndSigma_T_Ref(const nlohmann::json &material,
                                    const nlohmann::json &materials,
                                    const std::string &model,
                                    double &alpha_resistivity,
                                    double &T_ref,
                                    double &sigma_T_ref)
{
   const auto &material_name = material["name"].get<std::string>();

   if (material["conductivity"].contains("alpha_resistivity"))
   {
      alpha_resistivity =
          material["conductivity"]["alpha_resistivity"].get<double>();
   }
   else
   {
      alpha_resistivity =
          materials[material_name]["conductivity"][model]["alpha_resistivity"]
              .get<double>();
   }
   if (material["conductivity"].contains("T_ref"))
   {
      T_ref = material["conductivity"]["T_ref"].get<double>();
   }
   else
   {
      T_ref = materials[material_name]["conductivity"][model]["T_ref"]
                  .get<double>();
   }
   if (material["conductivity"].contains("sigma_T_ref"))
   {
      sigma_T_ref = material["conductivity"]["sigma_T_ref"].get<double>();
   }
   else
   {
      sigma_T_ref =
          materials[material_name]["conductivity"][model]["sigma_T_ref"]
              .get<double>();
   }
}

// Construct the conductivity coefficient
std::unique_ptr<mfem::Coefficient> constructConductivityCoeff(
    const nlohmann::json &component,
    const nlohmann::json &materials)
{
   std::unique_ptr<mfem::Coefficient>
       temp_coeff;  // temp=temporary, not temperature
   const auto &material = component["material"];  // set material

   /// If "material" is a string, it is interpreted to be the name of a
   /// material. We default to a conductivity of ///TODO: (insert default value
   /// here) unless there is a different value in the material library
   if (material.is_string())
   {
      const auto &material_name = material.get<std::string>();
      temp_coeff = constructConstantConductivityCoeff(material_name, materials);
   }
   else
   {
      const auto &material_name = material["name"].get<std::string>();

      // Aligns with newer format of the material library JSON structure
      if (material.contains("conductivity"))
      {
         const auto &sigma_model =
             material["conductivity"]["model"].get<std::string>();
         if (sigma_model == "constant")
         {
            temp_coeff =
                constructConstantConductivityCoeff(material_name, materials);
         }
         else if (sigma_model == "linear")
         {
            double alpha_resistivity;
            double T_ref;
            double sigma_T_ref;
            getAlphaAndT_RefAndSigma_T_Ref(material,
                                           materials,
                                           sigma_model,
                                           alpha_resistivity,
                                           T_ref,
                                           sigma_T_ref);
            temp_coeff = std::make_unique<LinearTempDepConductivityCoefficient>(
                alpha_resistivity, T_ref, sigma_T_ref);
         }
         else
         {
            std::string error_msg =
                "Unrecognized conductivity model for material \"";
            error_msg += material_name;
            error_msg += "\"!\n";
            throw mach::MachException(error_msg);
         }
      }
      else
      {
         temp_coeff =
             constructConstantConductivityCoeff(material_name, materials);
      }
   }
   return temp_coeff;
}

}  // anonymous namespace

namespace mach
{
double ConductivityCoefficient::Eval(mfem::ElementTransformation &trans,
                                     const mfem::IntegrationPoint &ip)
{
   return sigma.Eval(trans, ip);
}

double ConductivityCoefficient::Eval(mfem::ElementTransformation &trans,
                                     const mfem::IntegrationPoint &ip,
                                     double state)
{
   return sigma.Eval(trans, ip, state);
}

double ConductivityCoefficient::EvalStateDeriv(
    mfem::ElementTransformation &trans,
    const mfem::IntegrationPoint &ip,
    double state)
{
   return sigma.EvalStateDeriv(trans, ip, state);
}

double ConductivityCoefficient::EvalState2ndDeriv(
    mfem::ElementTransformation &trans,
    const mfem::IntegrationPoint &ip,
    const double state)
{
   return sigma.EvalState2ndDeriv(trans, ip, state);
}

void ConductivityCoefficient::EvalRevDiff(const double Q_bar,
                                          mfem::ElementTransformation &trans,
                                          const mfem::IntegrationPoint &ip,
                                          double state,
                                          mfem::DenseMatrix &PointMat_bar)
{
   sigma.EvalRevDiff(Q_bar, trans, ip, state, PointMat_bar);
}

/// TODO: Change sigma(std::make_unique<mfem::ConstantCoefficient>(58.14e6) line
/// IF the equivalent line... std::unique_ptr<mfem::Coefficient>
/// constructConstantConductivityCoeff( from earlier changes
ConductivityCoefficient::ConductivityCoefficient(
    const nlohmann::json &sigma_options,
    const nlohmann::json &materials)
 : sigma(std::make_unique<mfem::ConstantCoefficient>(58.14e6))
{
   /// loop over all components, construct a conductivity coefficient for each
   for (const auto &component : sigma_options["components"])
   {
      int attr = component.value("attr", -1);
      if (-1 != attr)
      {
         sigma.addCoefficient(attr,
                              constructConductivityCoeff(component, materials));
      }
      else
      {
         for (const auto &attribute : component["attrs"])
         {
            sigma.addCoefficient(
                attribute, constructConductivityCoeff(component, materials));
         }
      }
   }
}

}  // namespace mach

namespace
{
LinearTempDepConductivityCoefficient::LinearTempDepConductivityCoefficient(
    const double &alpha_resistivity,
    const double &T_ref,
    const double &sigma_T_ref)
 : alpha_resistivity(alpha_resistivity), T_ref(T_ref), sigma_T_ref(sigma_T_ref)

{ }

double LinearTempDepConductivityCoefficient::Eval(
    mfem::ElementTransformation &trans,
    const mfem::IntegrationPoint &ip,
    const double state)
{
   double T = state;  // assuming the state is the temperature
   // Evaluate the value for the conductivity sigma
   double sigma = sigma_T_ref / (1 + alpha_resistivity * (T - T_ref));
   return sigma;
}

double LinearTempDepConductivityCoefficient::EvalStateDeriv(
    mfem::ElementTransformation &trans,
    const mfem::IntegrationPoint &ip,
    const double state)
{
   double T = state;  // assuming the state is the temperature
   // Evaluate the derivative of sigma with respect to the state (temperature)
   double dsigmadT = (-sigma_T_ref * alpha_resistivity) /
                     std::pow(1 + alpha_resistivity * (T - T_ref), 2);
   return dsigmadT;
}

double LinearTempDepConductivityCoefficient::EvalState2ndDeriv(
    mfem::ElementTransformation &trans,
    const mfem::IntegrationPoint &ip,
    const double state)
{
   double T = state;  // assuming the state is the temperature
   // Evaluate the second derivative of sigma with respect to the state
   // (temperature)
   double d2sigmadT2 = (2 * sigma_T_ref * std::pow(alpha_resistivity, 2)) /
                       std::pow(1 + alpha_resistivity * (T - T_ref), 3);
   return d2sigmadT2;
}

}  // namespace