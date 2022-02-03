#include <memory>
#include <string>
#include <vector>

#include "mfem.hpp"
#include "nlohmann/json.hpp"

#include "reluctivity_coefficient.hpp"

namespace
{
/// permeability of free space
constexpr double mu_0 = 4e-7 * M_PI;

std::unique_ptr<mfem::Coefficient> constructReluctivityCoeff(
    const nlohmann::json &component,
    const nlohmann::json &materials)
{
   std::unique_ptr<mfem::Coefficient> temp_coeff;
   auto material = component["material"].get<std::string>();

   auto has_nonlinear =
       materials[material].contains("B") && materials[material].contains("H");
   if (component.contains("linear"))
   {
      auto linear = component["linear"].get<bool>();
      if (linear)
      {
         auto mu_r = materials[material]["mu_r"].get<double>();
         temp_coeff =
             std::make_unique<mfem::ConstantCoefficient>(1.0 / (mu_r * mu_0));
      }
      else
      {
         auto b = materials[material]["B"].get<std::vector<double>>();
         auto h = materials[material]["H"].get<std::vector<double>>();
         temp_coeff =
             std::make_unique<mach::NonlinearReluctivityCoefficient>(b, h);
      }
   }
   else
   {
      if (has_nonlinear)
      {
         auto b = materials[material]["B"].get<std::vector<double>>();
         auto h = materials[material]["H"].get<std::vector<double>>();
         temp_coeff =
             std::make_unique<mach::NonlinearReluctivityCoefficient>(b, h);
      }
      else
      {
         auto mu_r = materials[material]["mu_r"].get<double>();
         temp_coeff =
             std::make_unique<mfem::ConstantCoefficient>(1.0 / (mu_r * mu_0));
      }
   }
   return temp_coeff;
}

}  // anonymous namespace

namespace mach
{
double ReluctivityCoefficient::Eval(mfem::ElementTransformation &trans,
                                    const mfem::IntegrationPoint &ip)
{
   return nu.Eval(trans, ip);
}

double ReluctivityCoefficient::Eval(mfem::ElementTransformation &trans,
                                    const mfem::IntegrationPoint &ip,
                                    double state)
{
   return nu.Eval(trans, ip, state);
}

double ReluctivityCoefficient::EvalStateDeriv(
    mfem::ElementTransformation &trans,
    const mfem::IntegrationPoint &ip,
    double state)
{
   return nu.EvalStateDeriv(trans, ip, state);
}

double ReluctivityCoefficient::EvalState2ndDeriv(
    mfem::ElementTransformation &trans,
    const mfem::IntegrationPoint &ip,
    const double state)
{
   return nu.EvalState2ndDeriv(trans, ip, state);
}

void ReluctivityCoefficient::EvalRevDiff(const double Q_bar,
                                         mfem::ElementTransformation &trans,
                                         const mfem::IntegrationPoint &ip,
                                         mfem::DenseMatrix &PointMat_bar)
{
   nu.EvalRevDiff(Q_bar, trans, ip, PointMat_bar);
}

ReluctivityCoefficient::ReluctivityCoefficient(const nlohmann::json &nu_options,
                                               const nlohmann::json &materials)
 : nu(std::make_unique<mfem::ConstantCoefficient>(1.0 / mu_0))
{
   /// loop over all components, construct a reluctivity coefficient for each
   for (auto &component : nu_options["components"])
   {
      int attr = component.value("attr", -1);
      if (-1 != attr)
      {
         nu.addCoefficient(attr,
                           constructReluctivityCoeff(component, materials));
      }
      else
      {
         for (auto &attribute : component["attrs"])
         {
            nu.addCoefficient(attribute,
                              constructReluctivityCoeff(component, materials));
         }
      }
   }
}

}  // namespace mach
