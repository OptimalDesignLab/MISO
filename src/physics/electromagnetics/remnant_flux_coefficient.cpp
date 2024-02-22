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
class LinearTempDepRemnantFluxCoefficient : public miso::StateCoefficient
{
public:
   /// \brief Define a remnant flux model that is a linear function of
   /// temperature \param[in] alpha_B_r - the remanent flux temperature
   /// coefficient in %/deg C or %/K. Given by the manufacturer. \param[in]
   /// T_ref - reference temperature at which the remnant flux is given in
   /// either deg C or K. Given by the manufacturer. \param[in] B_r_T_ref - the
   /// remnant flux in Teslas at the reference temperature. Given by the
   /// manufacturer.
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

   /// \brief Evaluate the derivative of remnant flux with respsect to
   /// Temperature (T) in the element described by trans at the point ip. \note
   /// When this method is called, the caller must make sure that the
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
                    double state,
                    mfem::DenseMatrix &PointMat_bar) override
   { }

protected:
   // alpha_B_r, T_ref, and B_r_T_ref are needed for methods
   double alpha_B_r, T_ref, B_r_T_ref;
};

// Function to extract the values for alpha, T_ref, and B_r from the material
// library JSON structure
void getAlphaAndT_RefAndB_r_T_Ref(const nlohmann::json &material,
                                  double &alpha_B_r,
                                  double &T_ref,
                                  double &B_r_T_ref)
{
   // const auto &material_name = material["name"].get<std::string>();

   if (material.contains("alpha_B_r"))
   {
      alpha_B_r = material["alpha_B_r"].get<double>();
   }
   else
   {
      /// TODO: Change this default value for alpha_B_r as needed!
      alpha_B_r = -0.12;
   }
   if (material.contains("T_ref"))
   {
      T_ref = material["T_ref"].get<double>();
   }
   else
   {
      /// TODO: Change this default value for T_ref as needed!
      T_ref = 293.15;
   }
   if (material.contains("B_r_T_ref"))
   {
      B_r_T_ref = material["B_r_T_ref"].get<double>();
   }
   else
   {
      /// TODO: Change this default value for B_r_T_ref as needed!
      B_r_T_ref = 1.39;
   }
}

// Construct the remnant flux coefficient
std::unique_ptr<mfem::Coefficient> constructRemnantFluxCoeff(
    const nlohmann::json &material)
{
   std::unique_ptr<mfem::Coefficient>
       temp_coeff;  // temp=temporary, not temperature
   // const auto &material = component["material"]; // set material

   /// Assuming that the material is the Nd2Fe14B JSON structure from material
   /// library

   // const auto &material_name = material["name"].get<std::string>();

   double alpha_B_r;
   double T_ref;
   double B_r_T_ref;
   getAlphaAndT_RefAndB_r_T_Ref(material, alpha_B_r, T_ref, B_r_T_ref);
   temp_coeff = std::make_unique<LinearTempDepRemnantFluxCoefficient>(
       alpha_B_r, T_ref, B_r_T_ref);

   return temp_coeff;
}

}  // anonymous namespace

namespace miso
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

void RemnantFluxCoefficient::EvalRevDiff(const double Q_bar,
                                         mfem::ElementTransformation &trans,
                                         const mfem::IntegrationPoint &ip,
                                         double state,
                                         mfem::DenseMatrix &PointMat_bar)
{
   B_r.EvalRevDiff(Q_bar, trans, ip, state, PointMat_bar);
}

/// TODO: Change B_r(std::make_unique<mfem::ConstantCoefficient>(1.39) line IF
/// the equivalent line... std::unique_ptr<mfem::Coefficient>
/// constructConstantRemnantFluxCoeff( from earlier changes
RemnantFluxCoefficient::RemnantFluxCoefficient(const nlohmann::json &material)
 : B_r(std::make_unique<mfem::ConstantCoefficient>(1.39))
{
   // /// construct a remnant flux coefficient for the material
   // int attr = material.value("attr", -1);
   // if (-1 != attr)
   // {
   //    std::cout << "(-1 != attr)\n";
   //    B_r.addCoefficient(attr,
   //                      constructRemnantFluxCoeff(material));
   // }
   // else
   // {
   //    std::cout << "(-1 == attr)\n";
   //    for (const auto &attribute : material["attrs"])
   //    {
   //       std::cout << "Going through attributes\n";
   //       B_r.addCoefficient(attribute,
   //                         constructRemnantFluxCoeff(material));
   //    }
   // }

   B_r = constructRemnantFluxCoeff(material);
}

}  // namespace miso

namespace
{
LinearTempDepRemnantFluxCoefficient::LinearTempDepRemnantFluxCoefficient(
    const double &alpha_B_r,
    const double &T_ref,
    const double &B_r_T_ref)
 : alpha_B_r(alpha_B_r), T_ref(T_ref), B_r_T_ref(B_r_T_ref)

{ }

double LinearTempDepRemnantFluxCoefficient::Eval(
    mfem::ElementTransformation &trans,
    const mfem::IntegrationPoint &ip,
    const double state)
{
   double T = state;  // assuming the state is the temperature
   // Evaluate the value for the remnant flux B_r
   double B_r = B_r_T_ref * (1 + (alpha_B_r / 100) * (T - T_ref));
   // std::cout << "B_r = B_r_T_ref*(1+(alpha_B_r/100)*(T-T_ref)) =" <<
   // B_r_T_ref << "*" << "(1+(" << alpha_B_r << "/100)*(" << T << "-" << T_ref
   // << "))= " << B_r << "\n";

   // double B_r = B_r_T_ref * log(1 + exp((1 + (alpha_B_r / 100) * (T -
   // T_ref))));

   // if (B_r < 0.0)
   // {
   //    return 0.0;
   // }

   return B_r;
}

double LinearTempDepRemnantFluxCoefficient::EvalStateDeriv(
    mfem::ElementTransformation &trans,
    const mfem::IntegrationPoint &ip,
    const double state)
{
   // double T=state; // assuming the state is the temperature
   // Evaluate the derivative of B_r with respect to the state (temperature)
   double dB_rdT = B_r_T_ref * (alpha_B_r / 100);
   // double dB_rdT = (alpha_B_r/100 * B_r_T_ref)*(1 - 1/(1 + exp((1 +
   // (alpha_B_r / 100) * (T - T_ref)))));
   return dB_rdT;
}

double LinearTempDepRemnantFluxCoefficient::EvalState2ndDeriv(
    mfem::ElementTransformation &trans,
    const mfem::IntegrationPoint &ip,
    const double state)
{
   //    double T=state; // assuming the state is the temperature
   // Evaluate the second derivative of B_r with respect to the state
   // (temperature)
   double d2B_rdT2 = 0;
   return d2B_rdT2;
}

}  // namespace