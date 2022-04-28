#include <string>

#include "mfem.hpp"
#include "nlohmann/json.hpp"

#include "mach_input.hpp"
#include "mfem_common_integ.hpp"
#include "thermal_integ.hpp"

#include "thermal_residual.hpp"

namespace
{
std::unique_ptr<mach::MeshDependentCoefficient> constructMaterialCoefficient(
    const std::string &name,
    const nlohmann::json &options,
    const nlohmann::json &materials,
    double default_val = 0.0)
{
   auto material_coeff = std::make_unique<mach::MeshDependentCoefficient>();
   /// loop over all components, construct coeff for each
   for (auto &component : options["components"])
   {
      int attr = component.value("attr", -1);

      const auto &material = component["material"].get<std::string>();
      double val = materials[material].value(name, default_val);

      if (-1 != attr)
      {
         auto coeff = std::make_unique<mfem::ConstantCoefficient>(val);
         material_coeff->addCoefficient(attr, move(coeff));
      }
      else
      {
         for (auto &attribute : component["attrs"])
         {
            auto coeff = std::make_unique<mfem::ConstantCoefficient>(val);
            material_coeff->addCoefficient(attribute, move(coeff));
         }
      }
   }
   return material_coeff;
}

}  // anonymous namespace

namespace mach
{
int getSize(const ThermalResidual &residual) { return getSize(residual.res); }

void setInputs(ThermalResidual &residual, const mach::MachInputs &inputs)
{
   setInputs(residual.res, inputs);
}

void setOptions(ThermalResidual &residual, const nlohmann::json &options)
{
   setOptions(residual.res, options);
}

void evaluate(ThermalResidual &residual,
              const mach::MachInputs &inputs,
              mfem::Vector &res_vec)
{
   evaluate(residual.res, inputs, res_vec);
}

void linearize(ThermalResidual &residual, const mach::MachInputs &inputs)
{
   linearize(residual.res, inputs);
}

mfem::Operator &getJacobian(ThermalResidual &residual,
                            const mach::MachInputs &inputs,
                            const std::string &wrt)
{
   return getJacobian(residual.res, inputs, wrt);
}

mfem::Operator &getJacobianTranspose(ThermalResidual &residual,
                                     const mach::MachInputs &inputs,
                                     const std::string &wrt)
{
   return getJacobianTranspose(residual.res, inputs, wrt);
}

void setUpAdjointSystem(ThermalResidual &residual,
                        mfem::Solver &adj_solver,
                        const mach::MachInputs &inputs,
                        mfem::Vector &state_bar,
                        mfem::Vector &adjoint)
{
   setUpAdjointSystem(residual.res, adj_solver, inputs, state_bar, adjoint);
}

double jacobianVectorProduct(ThermalResidual &residual,
                             const mfem::Vector &wrt_dot,
                             const std::string &wrt)
{
   auto res_dot = jacobianVectorProduct(residual.res, wrt_dot, wrt);
   return res_dot;
}

void jacobianVectorProduct(ThermalResidual &residual,
                           const mfem::Vector &wrt_dot,
                           const std::string &wrt,
                           mfem::Vector &res_dot)
{
   jacobianVectorProduct(residual.res, wrt_dot, wrt, res_dot);
}

double vectorJacobianProduct(ThermalResidual &residual,
                             const mfem::Vector &res_bar,
                             const std::string &wrt)
{
   auto wrt_bar = vectorJacobianProduct(residual.res, res_bar, wrt);
   return wrt_bar;
}

void vectorJacobianProduct(ThermalResidual &residual,
                           const mfem::Vector &res_bar,
                           const std::string &wrt,
                           mfem::Vector &wrt_bar)
{
   vectorJacobianProduct(residual.res, res_bar, wrt, wrt_bar);
}

mfem::Solver *getPreconditioner(ThermalResidual &residual)
{
   return residual.prec.get();
}

ThermalResidual::ThermalResidual(
    mfem::ParFiniteElementSpace &fes,
    std::map<std::string, FiniteElementState> &fields,
    const nlohmann::json &options,
    const nlohmann::json &materials)
 : res(fes, fields),
   kappa(constructMaterialCoefficient("kappa", options, materials)),
   rho(constructMaterialCoefficient("rho", options, materials)),
   cv(constructMaterialCoefficient("cv", options, materials)),
   prec(constructPreconditioner(fes, options["lin-prec"]))

{
   const auto &basis_type =
       options["space-dis"]["basis-type"].get<std::string>();
   if (basis_type != "H1" && basis_type != "h1" && basis_type != "CG" &&
       basis_type != "cg")
   {
      throw MachException(
          "Thermal residual currently only supports H1 state field!\n");
   }

   res.addDomainIntegrator(new mfem::DiffusionIntegrator(*kappa));

   if (options.contains("bcs"))
   {
      auto &bcs = options["bcs"];

      // convection heat transfer boundary condition
      if (bcs.contains("convection"))
      {
         const auto &bdr_attr_marker =
             bcs["convection"].get<std::vector<int>>();

         res.addBdrFaceIntegrator(new ConvectionBCIntegrator,
                                  bdr_attr_marker);
      }
   }
}

}  // namespace mach
