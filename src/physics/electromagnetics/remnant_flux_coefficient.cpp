#include <cmath>
#include <memory>
#include <string>
#include <vector>

#include "mfem.hpp"
#include "nlohmann/json.hpp"
#include "tinysplinecxx.h"

#include "remnant_flux_coefficient.hpp"
#include "utils.hpp"

namespace
{
class LinearTempDepRemnantFluxCoefficient : public mach::StateCoefficient
{
public:
   /// \brief Define a remnant flux model that is a linear function of temperature
   /// \param[in] alpha_B_r - the remanent flux temperature coefficient in %/deg C or %/K. Given by the manufacturer.
   /// \param[in] T_ref - reference temperature at which the remnant flux is given in either deg C or K. Given by the manufacturer.
   /// \param[in] B_r_T_ref - the remnant flux in Teslas at the reference temperature. Given by the manufacturer.
   LinearTempDepRemnantFluxCoefficient(const double &alpha_B_r,
                                        const double &T_ref,
                                        const double &B_r_T_ref);

   /// \brief Evaluate the remnant flux in the element described by trans at the
   /// point ip.
   /// \note When this method is called, the caller must make sure that the
   /// IntegrationPoint associated with trans is the same as ip. This can be
   /// achieved by calling trans.SetIntPoint(&ip).
   double Eval(mfem::ElementTransformation &trans,
               const mfem::IntegrationPoint &ip,
               double state) override;

   /// \brief Evaluate the derivative of remnant flux with respsect to Temperature (T) in the
   /// element described by trans at the point ip.
   /// \note When this method is called, the caller must make sure that the
   /// IntegrationPoint associated with trans is the same as ip. This can be
   /// achieved by calling trans.SetIntPoint(&ip).
   double EvalStateDeriv(mfem::ElementTransformation &trans,
                         const mfem::IntegrationPoint &ip,
                         double state) override;

   /// \brief Evaluate the remnant flux in the element described by trans at the
   /// point ip.
   /// \note When this method is called, the caller must make sure that the
   /// IntegrationPoint associated with trans is the same as ip. This can be
   /// achieved by calling trans.SetIntPoint(&ip).
/// TODO: Determine if this is necessary to keep
   double EvalState2ndDeriv(mfem::ElementTransformation &trans,
                            const mfem::IntegrationPoint &ip,
                            const double state) override;

/// TODO: Adapt EvalRevDiff as needed for remnant flux
   void EvalRevDiff(const double Q_bar,
                    mfem::ElementTransformation &trans,
                    const mfem::IntegrationPoint &ip,
                    mfem::DenseMatrix &PointMat_bar) override
   { }

protected:
   // alpha_B_r, T_ref, and B_r_T_ref are needed for methods
   double alpha_B_r, T_ref, B_r_T_ref;
};

/// Serves as a default value for the remnant flux
std::unique_ptr<mfem::Coefficient> constructConstantRemnantFluxCoeff(
    const std::string &material_name,
    const nlohmann::json &materials)
{
   /// TODO: Decide if want to change default value of B_r to 1 or leave it as 1.39
   auto B_r = materials[material_name].value("B_r", 1.39);
   return std::make_unique<mfem::ConstantCoefficient>(B_r);
}

// Function to extract the values for alpha, T_ref, and B_r from the material library JSON structure
void getAlphaAndT_RefAndB_r_T_Ref(const nlohmann::json &material,
                          const nlohmann::json &materials,
                          double &alpha_B_r,
                          double &T_ref,
                          double &B_r_T_ref)
{
   const auto &material_name = material["name"].get<std::string>();

   if (material.contains("alpha_B_r"))
   {
      alpha_B_r = material["alpha_B_r"].get<double>();
   }
   else if (materials[material_name].contains("alpha_B_r"))
   {
      alpha_B_r = materials[material_name]["alpha_B_r"]
                .get<double>();
   }
   else
   {
      ///TODO: Change this default value for alpha_B_r as needed!  
      alpha_B_r = -0.12; 
   }
   if (material.contains("T_ref"))
   {
      T_ref = material["T_ref"].get<double>();
   }
   else if (materials[material_name].contains("T_ref"))
   {
      T_ref = materials[material_name]["T_ref"]
                .get<double>();
   }
   else
   {
      ///TODO: Change this default value for T_ref as needed!  
      T_ref = 20.0; 
   }   
   if (material.contains("B_r_T_ref"))
   {
      B_r_T_ref = material["B_r_T_ref"].get<double>();
   }
   else if (materials[material_name].contains("B_r_T_ref"))
   {
      B_r_T_ref = materials[material_name]["B_r_T_ref"]
                .get<double>();
   }
   else
   {
      ///TODO: Change this default value for B_r_T_ref as needed!  
      B_r_T_ref = 1.39; 
   }
}

// Construct the remnant flux coefficient
std::unique_ptr<mfem::Coefficient> constructRemnantFluxCoeff(
    const nlohmann::json &component,
    const nlohmann::json &materials)
{
   std::unique_ptr<mfem::Coefficient> temp_coeff; // temp=temporary, not temperature
   const auto &material = component["material"]; // set material

   /// If "material" is a string, it is interpreted to be the name of a 
   /// material. We default to a remnant flux of ///TODO: (insert default value here) unless
   /// there is a different value in the material library
   if (material.is_string())
   {
      const auto &material_name = material.get<std::string>();
      temp_coeff = constructConstantRemnantFluxCoeff(material_name, materials);
   }
   else
   {
      const auto &material_name = material["name"].get<std::string>();
      
        double alpha_B_r;
        double T_ref;
        double B_r_T_ref;
        getAlphaAndT_RefAndB_r_T_Ref(material, materials, alpha_B_r, T_ref,B_r_T_ref);
        temp_coeff = std::make_unique<LinearTempDepRemnantFluxCoefficient>(
            alpha_B_r, T_ref, B_r_T_ref);
   }
   return temp_coeff;
}

}  // anonymous namespace

namespace mach
{
double RemnantFluxCoefficient::Eval(mfem::ElementTransformation &trans,
                                    const mfem::IntegrationPoint &ip)
{
   return B_r.Eval(trans, ip);
}

double RemnantFluxCoefficient::Eval(mfem::ElementTransformation &trans,
                                    const mfem::IntegrationPoint &ip,
                                    double state)
{
   return B_r.Eval(trans, ip, state);
}

double RemnantFluxCoefficient::EvalStateDeriv(
    mfem::ElementTransformation &trans,
    const mfem::IntegrationPoint &ip,
    double state)
{
   return B_r.EvalStateDeriv(trans, ip, state);
}

double RemnantFluxCoefficient::EvalState2ndDeriv(
    mfem::ElementTransformation &trans,
    const mfem::IntegrationPoint &ip,
    const double state)
{
   return B_r.EvalState2ndDeriv(trans, ip, state);
}

/// TODO: Adapt if keeping, remove if not
void RemnantFluxCoefficient::EvalRevDiff(const double Q_bar,
                                         mfem::ElementTransformation &trans,
                                         const mfem::IntegrationPoint &ip,
                                         mfem::DenseMatrix &PointMat_bar)
{
   B_r.EvalRevDiff(Q_bar, trans, ip, PointMat_bar);
}

/// TODO: Change B_r(std::make_unique<mfem::ConstantCoefficient>(1.39) line IF the equivalent line...
/// std::unique_ptr<mfem::Coefficient> constructConstantRemnantFluxCoeff( from earlier changes
RemnantFluxCoefficient::RemnantFluxCoefficient(const nlohmann::json &B_r_options,
                                               const nlohmann::json &materials)
 : B_r(std::make_unique<mfem::ConstantCoefficient>(1.39)) 
{
   /// loop over all components, construct a remnant flux coefficient for each
   for (const auto &component : B_r_options["components"])
   {
      int attr = component.value("attr", -1);
      if (-1 != attr)
      {
         B_r.addCoefficient(attr,
                           constructRemnantFluxCoeff(component, materials));
      }
      else
      {
         for (const auto &attribute : component["attrs"])
         {
            B_r.addCoefficient(attribute,
                              constructRemnantFluxCoeff(component, materials));
         }
      }
   }
}

}  // namespace mach

namespace
{
LinearTempDepRemnantFluxCoefficient::LinearTempDepRemnantFluxCoefficient(
   const double &alpha_B_r,
   const double &T_ref,
   const double &B_r_T_ref)
 : alpha_B_r(alpha_B_r),
   T_ref(T_ref),
   B_r_T_ref(B_r_T_ref)

{
  

}

double LinearTempDepRemnantFluxCoefficient::Eval(
   mfem::ElementTransformation &trans,
   const mfem::IntegrationPoint &ip,
   const double state)
{
   double T=state; // assuming the state is the temperature
   // Evaluate the value for the remnant flux B_r
   double B_r = B_r_T_ref*(1+(alpha_B_r/100)*(T-T_ref));
   return B_r;

}

double LinearTempDepRemnantFluxCoefficient::EvalStateDeriv(
    mfem::ElementTransformation &trans,
    const mfem::IntegrationPoint &ip,
    const double state)
{
//    double T=state; // assuming the state is the temperature
   // Evaluate the derivative of B_r with respect to the state (temperature)
   double dB_rdT = B_r_T_ref*(alpha_B_r/100);
   return dB_rdT;
}

double LinearTempDepRemnantFluxCoefficient::EvalState2ndDeriv(
    mfem::ElementTransformation &trans,
    const mfem::IntegrationPoint &ip,
    const double state)
{
//    double T=state; // assuming the state is the temperature
   // Evaluate the second derivative of B_r with respect to the state (temperature)
   double d2B_rdT2 = 0;
   return d2B_rdT2;
}

}