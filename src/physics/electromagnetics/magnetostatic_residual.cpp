#include <memory>
#include <string>

#include "adept.h"
#include "electromag_integ.hpp"
#include "mach_linearform.hpp"
#include "mfem.hpp"
#include "nlohmann/json.hpp"

#include "mach_input.hpp"
#include "magnetostatic_load.hpp"

#include "magnetostatic_residual.hpp"
#include "utils.hpp"

namespace
{
std::unique_ptr<mfem::Solver> constructPreconditioner(
    mfem::ParFiniteElementSpace &fes,
    const nlohmann::json &prec_options)
{
   auto prec_type = prec_options["type"].get<std::string>();
   if (prec_type == "hypreams")
   {
      auto ams = std::make_unique<mfem::HypreAMS>(&fes);
      ams->SetPrintLevel(prec_options["printlevel"].get<int>());
      ams->SetSingularProblem();
      return ams;
   }
   else if (prec_type == "hypreboomeramg")
   {
      auto amg = std::make_unique<mfem::HypreBoomerAMG>();
      amg->SetPrintLevel(prec_options["printlevel"].get<int>());
      return amg;
   }
   return nullptr;
}

}  // namespace

namespace mach
{
int getSize(const MagnetostaticResidual &residual)
{
   return getSize(residual.res);
}

void setInputs(MagnetostaticResidual &residual, const mach::MachInputs &inputs)
{
   setInputs(residual.res, inputs);
   setInputs(*residual.load, inputs);
}

void setOptions(MagnetostaticResidual &residual, const nlohmann::json &options)
{
   setOptions(residual.res, options);
   setOptions(*residual.load, options);
}

void evaluate(MagnetostaticResidual &residual,
              const mach::MachInputs &inputs,
              mfem::Vector &res_vec)
{
   evaluate(residual.res, inputs, res_vec);
   setInputs(*residual.load, inputs);
   addLoad(*residual.load, res_vec);

   // mfem::Vector state;
   // setVectorFromInputs(inputs, "state", state);
   // const auto &ess_tdofs = residual.res.getEssentialDofs();
   // for (int i = 0; i < ess_tdofs.Size(); ++i)
   // {
   //    res_vec(ess_tdofs[i]) = state(ess_tdofs[i]);
   // }
}

void linearize(MagnetostaticResidual &residual, const mach::MachInputs &inputs)
{
   linearize(residual.res, inputs);
}

mfem::Operator &getJacobian(MagnetostaticResidual &residual,
                            const mach::MachInputs &inputs,
                            const std::string &wrt)
{
   return getJacobian(residual.res, inputs, wrt);
}

mfem::Operator &getJacobianTranspose(MagnetostaticResidual &residual,
                                     const mach::MachInputs &inputs,
                                     const std::string &wrt)
{
   return getJacobianTranspose(residual.res, inputs, wrt);
}

void setUpAdjointSystem(MagnetostaticResidual &residual,
                        mfem::Solver &adj_solver,
                        const mach::MachInputs &inputs,
                        mfem::Vector &state_bar,
                        mfem::Vector &adjoint)
{
   setUpAdjointSystem(residual.res, adj_solver, inputs, state_bar, adjoint);
}

double jacobianVectorProduct(MagnetostaticResidual &residual,
                             const mfem::Vector &wrt_dot,
                             const std::string &wrt)
{
   auto res_dot = jacobianVectorProduct(residual.res, wrt_dot, wrt);
   res_dot += jacobianVectorProduct(*residual.load, wrt_dot, wrt);
   return res_dot;
}

void jacobianVectorProduct(MagnetostaticResidual &residual,
                           const mfem::Vector &wrt_dot,
                           const std::string &wrt,
                           mfem::Vector &res_dot)
{
   jacobianVectorProduct(residual.res, wrt_dot, wrt, res_dot);
   jacobianVectorProduct(*residual.load, wrt_dot, wrt, res_dot);
}

double vectorJacobianProduct(MagnetostaticResidual &residual,
                             const mfem::Vector &res_bar,
                             const std::string &wrt)
{
   auto wrt_bar = vectorJacobianProduct(residual.res, res_bar, wrt);
   wrt_bar += vectorJacobianProduct(*residual.load, res_bar, wrt);
   return wrt_bar;
}

void vectorJacobianProduct(MagnetostaticResidual &residual,
                           const mfem::Vector &res_bar,
                           const std::string &wrt,
                           mfem::Vector &wrt_bar)
{
   vectorJacobianProduct(residual.res, res_bar, wrt, wrt_bar);
   vectorJacobianProduct(*residual.load, res_bar, wrt, wrt_bar);
}

mfem::Solver *getPreconditioner(MagnetostaticResidual &residual)
{
   return residual.prec.get();
}

MagnetostaticResidual::MagnetostaticResidual(
    adept::Stack &diff_stack,
    mfem::ParFiniteElementSpace &fes,
    std::map<std::string, FiniteElementState> &fields,
    const nlohmann::json &options,
    const nlohmann::json &materials,
    StateCoefficient &nu)
 : res(fes, fields),
   // load(diff_stack, fes, fields, options, materials, nu),
   prec(constructPreconditioner(fes, options["lin-prec"]))
{
   auto *mesh = fes.GetParMesh();
   auto space_dim = mesh->SpaceDimension();
   if (space_dim == 3)
   {
      res.addDomainIntegrator(new CurlCurlNLFIntegrator(nu));
      load = std::make_unique<MachLoad>(
          MagnetostaticLoad(diff_stack, fes, fields, options, materials, nu));
   }
   else if (space_dim == 2)
   {
      res.addDomainIntegrator(new NonlinearDiffusionIntegrator(nu));

      MachLinearForm linear_form(fes, fields);
      if (options.contains("current"))
      {
         current_coeff = std::make_unique<CurrentDensityCoefficient2D>(
             diff_stack, options["current"]);
         linear_form.addDomainIntegrator(
             new mfem::DomainLFIntegrator(*current_coeff));
      }
      if (options.contains("magnets"))
      {
         mag_coeff = std::make_unique<MagnetizationCoefficient>(
             diff_stack, options["magnets"], materials, 2);
         nuM = std::make_unique<mfem::ScalarVectorProductCoefficient>(
             nu, *mag_coeff);

         linear_form.addDomainIntegrator(
             new MagnetizationSource2DIntegrator(*nuM, 1.0));
      }

      load = std::make_unique<MachLoad>(std::move(linear_form));
   }
   else
   {
      throw MachException(
          "Invalid mesh dimension for Magnetostatic Residual!\n");
   }
}

}  // namespace mach
