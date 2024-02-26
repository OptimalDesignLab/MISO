#include <string>

#include "mfem.hpp"
#include "nlohmann/json.hpp"

#include "coefficient.hpp"
#include "miso_input.hpp"
#include "mfem_common_integ.hpp"
#include "thermal_integ.hpp"

#include "thermal_residual.hpp"

namespace miso
{
int getSize(const ThermalResidual &residual) { return getSize(residual.res); }

void setInputs(ThermalResidual &residual, const miso::MISOInputs &inputs)
{
   setInputs(residual.res, inputs);

   setVectorFromInputs(inputs, "thermal_load", residual.load);
   if (residual.load.Size() != 0)
   {
      residual.load.SetSubVector(residual.res.getEssentialDofs(), 0.0);
   }
}

void setOptions(ThermalResidual &residual, const nlohmann::json &options)
{
   setOptions(residual.res, options);
}

void evaluate(ThermalResidual &residual,
              const miso::MISOInputs &inputs,
              mfem::Vector &res_vec)
{
   evaluate(residual.res, inputs, res_vec);

   if (residual.load.Size() == res_vec.Size())
   {
      res_vec.Add(-1.0, residual.load);
   }
}

void linearize(ThermalResidual &residual, const miso::MISOInputs &inputs)
{
   linearize(residual.res, inputs);
}

mfem::Operator &getJacobian(ThermalResidual &residual,
                            const miso::MISOInputs &inputs,
                            const std::string &wrt)
{
   return getJacobian(residual.res, inputs, wrt);
}

mfem::Operator &getJacobianTranspose(ThermalResidual &residual,
                                     const miso::MISOInputs &inputs,
                                     const std::string &wrt)
{
   return getJacobianTranspose(residual.res, inputs, wrt);
}

void setUpAdjointSystem(ThermalResidual &residual,
                        mfem::Solver &adj_solver,
                        const miso::MISOInputs &inputs,
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
   kappa(
       constructMaterialCoefficient("kappa", options["components"], materials)),
   rho(constructMaterialCoefficient("rho", options["components"], materials)),
   cv(constructMaterialCoefficient("cv", options["components"], materials)),
   prec(constructPreconditioner(fes, options["lin-prec"]))

{
   const auto &basis_type =
       options["space-dis"]["basis-type"].get<std::string>();
   if (basis_type != "H1" && basis_type != "h1" && basis_type != "CG" &&
       basis_type != "cg")
   {
      throw MISOException(
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

         res.addBdrFaceIntegrator(new ConvectionBCIntegrator, bdr_attr_marker);
      }
   }
}

}  // namespace miso
