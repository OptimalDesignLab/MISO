#include <cmath>
#include <memory>
#include <string>
#include <vector>

#include "mfem.hpp"
#include "nlohmann/json.hpp"
#include "tinysplinecxx.h"

#include "conductivity_coefficient.hpp"
#include "utils.hpp"

/// TODO: Add/remove above includes as needed
namespace
{
/// TODO: If needed, add in necessary global variables

class LinearTempDepConductivityCoefficient : public mach::StateCoefficient
{
public:
   /// \brief Define a conductivity model that is a linear function of temperature
   /// \param[in] alpha_resistivity - temperature dependent resistivity coefficient (TODO: ensure double type is correct)
   /// \param[in] T_ref - reference temperature (TODO: ensure double type is correct)
   /// \param[in] sigma_T_ref - the conductivity at the reference temperature (TODO: ensure double type is correct)
   LinearTempDepConductivityCoefficient(double &alpha_resistivity,
                                        double &T_ref,
                                        double &sigma_T_ref);

   /// \brief Evaluate the conductivity in the element described by trans at the
   /// point ip.
   /// \note When this method is called, the caller must make sure that the
   /// IntegrationPoint associated with trans is the same as ip. This can be
   /// achieved by calling trans.SetIntPoint(&ip).
   double Eval(mfem::ElementTransformation &trans,
               const mfem::IntegrationPoint &ip,
               double state) override;

   /// \brief Evaluate the derivative of conductivity with respsect to Temperature (T) in the
   /// element described by trans at the point ip.
   /// \note When this method is called, the caller must make sure that the
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

/// TODO: Adapt EvalRevDiff as needed for conductivity
   void EvalRevDiff(const double Q_bar,
                    mfem::ElementTransformation &trans,
                    const mfem::IntegrationPoint &ip,
                    mfem::DenseMatrix &PointMat_bar) override
   { }

protected:
   // alpha_resistivity, T_ref, and sigma_T_ref are needed for methods
   double alpha_resistivity, T_ref, sigma_T_ref;

/// TODO: Add in any more protected class members that will be useful (protected meaning child classes can access too, but other classes cannot)
   
};

/// Serves as a default value for the conductivity
std::unique_ptr<mfem::Coefficient> constructConstantConductivityCoeff(
    const std::string &material_name,
    const nlohmann::json &materials)
{
   /// TODO: Make sure agrees with material library json structure at the end of the day
   /// TODO: Change default value of sigma to something more appropriate than 1
   auto sigma = materials[material_name].value("sigma", 1.0);
   return std::make_unique<mfem::ConstantCoefficient>(sigma);
}

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
      alpha_resistivity = material["conductivity"]["alpha_resistivity"].get<double>();
   }
   else
   {
      alpha_resistivity = materials[material_name]["conductivity"][model]["alpha_resistivity"]
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
      T_ref = material["conductivity"]["sigma_T_ref"].get<double>();
   }
   else
   {
      T_ref = materials[material_name]["conductivity"][model]["sigma_T_ref"]
                .get<double>();
   }
}

std::unique_ptr<mfem::Coefficient> constructConductivityCoeff(
    const nlohmann::json &component,
    const nlohmann::json &materials)
{
   std::unique_ptr<mfem::Coefficient> temp_coeff; // temp=temporary, not temperature
   const auto &material = component["material"];

   /// If "material" is a string, it is interpreted to be the name of a 
   /// material. We default to a conductivity of ///TODO: (insert default value here) unless
   /// there is a different value in the material library
   if (material.is_string())
   {
      const auto &material_name = material.get<std::string>();
      temp_coeff = constructConstantConductivityCoeff(material_name, materials);
   }
   else
   {
      const auto &material_name = material["name"].get<std::string>();

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
            getAlphaAndT_RefAndSigma_T_Ref(material, materials, sigma_model, alpha_resistivity, T_ref,sigma_T_ref);
            temp_coeff = std::make_unique<LinearTempDepConductivityCoefficient>(
                alpha_resistivity, T_ref);
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
         temp_coeff = constructConstantConductivityCoeff(material_name, materials);
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

/// TODO: Adapt if keeping, remove if not
void ConductivityCoefficient::EvalRevDiff(const double Q_bar,
                                         mfem::ElementTransformation &trans,
                                         const mfem::IntegrationPoint &ip,
                                         mfem::DenseMatrix &PointMat_bar)
{
   sigma.EvalRevDiff(Q_bar, trans, ip, PointMat_bar);
}

/// TODO: Change sigma(std::make_unique<mfem::ConstantCoefficient>(1.0) line IF the equivalent line...
/// std::unique_ptr<mfem::Coefficient> constructConstantConductivityCoeff( from earlier changes
ConductivityCoefficient::ConductivityCoefficient(const nlohmann::json &sigma_options,
                                               const nlohmann::json &materials)
 : sigma(std::make_unique<mfem::ConstantCoefficient>(1.0)) 
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
            sigma.addCoefficient(attribute,
                              constructConductivityCoeff(component, materials));
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
 : alpha_resistivity(std::make_unique<mfem::ConstantCoefficient>(alpha_resistivity)),
   T_ref(std::make_unique<mfem::ConstantCoefficient>(T_ref)),
   sigma_T_ref(std::make_unique<mfem::ConstantCoefficient>(sigma_T_ref))

///TODO: As needed, add in more definitions of protected class members here
{
  
///TODO: As needed, add in calculations of protected class members here

}

double LinearTempDepConductivityCoefficient::Eval(
   mfem::ElementTransformation &trans,
   const mfem::IntegrationPoint &ip,
   const double state)
{
   ///TODO: As needed, utilize logic of protected class members to eval sigma
    
   double sigma = sigma_T_ref/(1+alpha_resistivity*(state-T_ref));
   return sigma;

}

double LinearTempDepConductivityCoefficient::EvalStateDeriv(
    mfem::ElementTransformation &trans,
    const mfem::IntegrationPoint &ip,
    const double state)
{
   double dsigmadT = (-sigma_T_ref*alpha_resistivity)/std::pow(1+alpha_resistivity*(state-T_ref),2);
   return dsigmadT;
}

double LinearTempDepConductivityCoefficient::EvalState2ndDeriv(
    mfem::ElementTransformation &trans,
    const mfem::IntegrationPoint &ip,
    const double state)
{
   double d2sigmadT2 = (2*sigma_T_ref*std::pow(alpha_resistivity,2))/std::pow(1+alpha_resistivity*(state-T_ref),3);
   return d2sigmadT2;
}

///TODO: is there a need to code EvalRevDiff method here?

}