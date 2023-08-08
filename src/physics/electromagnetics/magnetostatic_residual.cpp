#include <memory>
#include <string>

#include "adept.h"
#include "electromag_integ.hpp"
#include "mach_linearform.hpp"
#include "mach_load.hpp"
#include "mfem.hpp"
#include "nlohmann/json.hpp"

#include "mach_input.hpp"
#include "magnetostatic_load.hpp"
#include "mfem_common_integ.hpp"
#include "utils.hpp"

#include "magnetostatic_residual.hpp"

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
   else if (prec_type == "hypreilu")
   {
      auto ilu = std::make_unique<mfem::HypreILU>();
      HYPRE_ILUSetType(*ilu, prec_options["ilu-type"]);
      HYPRE_ILUSetLevelOfFill(*ilu, prec_options["lev-fill"]);
      HYPRE_ILUSetLocalReordering(*ilu, prec_options["ilu-reorder"]);
      HYPRE_ILUSetPrintLevel(*ilu, prec_options["printlevel"]);
   }
   return nullptr;
}

std::vector<int> getCurrentAttributes(const nlohmann::json &options)
{
   std::vector<int> attributes;
   for (const auto &group : options["current"])
   {
      for (const auto &source : group)
      {
         auto attrs = source.get<std::vector<int>>();
         attributes.insert(attributes.end(), attrs.begin(), attrs.end());
      }
   }
   return attributes;
}

std::vector<int> getMagnetAttributes(const nlohmann::json &options)
{
   std::vector<int> attributes;
   for (const auto &group : options["magnets"])
   {
      for (const auto &source : group)
      {
         auto attrs = source.get<std::vector<int>>();
         attributes.insert(attributes.end(), attrs.begin(), attrs.end());
      }
   }
   return attributes;
}

}  // anonymous namespace

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
   if (residual.current_coeff != nullptr)
   {
      setInputs(*residual.current_coeff, inputs);
   }
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
   if (residual.current_coeff != nullptr)
   {
      setInputs(*residual.current_coeff, inputs);
   }
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

void finalizeAdjointSystem(MagnetostaticResidual &residual,
                           mfem::Solver &adj_solver,
                           const mach::MachInputs &inputs,
                           mfem::Vector &state_bar,
                           mfem::Vector &adjoint)
{
   finalizeAdjointSystem(residual.res, adj_solver, inputs, state_bar, adjoint);
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
   // if wrt starts with prefix "current_density:"
   if (wrt.rfind("current_density:", 0) == 0)
   {
      residual.current_coeff->cacheCurrentDensity();
      residual.current_coeff->zeroCurrentDensity();

      MachInputs inputs{{wrt, 1.0}};
      setInputs(*residual.current_coeff, inputs);

      residual.scratch.SetSize(res_dot.Size());
      residual.scratch = 0.0;
      addLoad(*residual.load, residual.scratch);
      residual.current_coeff->resetCurrentDensityFromCache();

      residual.scratch.SetSubVector(residual.res.getEssentialDofs(), 0.0);
      res_dot.Add(wrt_dot(0), residual.scratch);
      return;
   }
   jacobianVectorProduct(residual.res, wrt_dot, wrt, res_dot);
   jacobianVectorProduct(*residual.load, wrt_dot, wrt, res_dot);
}

double vectorJacobianProduct(MagnetostaticResidual &residual,
                             const mfem::Vector &res_bar,
                             const std::string &wrt)
{
   // if wrt starts with prefix "current_density:"
   if (wrt.rfind("current_density:", 0) == 0)
   {
      residual.current_coeff->cacheCurrentDensity();
      residual.current_coeff->zeroCurrentDensity();

      MachInputs inputs{{wrt, 1.0}};
      setInputs(*residual.current_coeff, inputs);

      residual.scratch.SetSize(res_bar.Size());
      residual.scratch = 0.0;
      addLoad(*residual.load, residual.scratch);
      residual.current_coeff->resetCurrentDensityFromCache();

      residual.scratch.SetSubVector(residual.res.getEssentialDofs(), 0.0);
      return residual.scratch * res_bar;
   }
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
   g(std::make_unique<mfem::GridFunctionCoefficient>(
       &fields.at("dirichlet_bc").gridFunc())),
   // load(diff_stack, fes, fields, options, materials, nu),
   prec(constructPreconditioner(fes, options["lin-prec"]))
{
   auto *mesh = fes.GetParMesh();
   auto space_dim = mesh->SpaceDimension();
   // auto space_dim = mesh->Dimension();
   if (space_dim == 3)
   {
      mesh->RemoveInternalBoundaries();

      res.addDomainIntegrator(new CurlCurlNLFIntegrator(nu));
      load = std::make_unique<MachLoad>(
          MagnetostaticLoad(diff_stack, fes, fields, options, materials, nu));
   }
   else if (space_dim == 2)
   {
      res.addDomainIntegrator(new NonlinearDiffusionIntegrator(nu));

      const auto &basis_type =
          options["space-dis"]["basis-type"].get<std::string>();

      if (basis_type == "L2" || basis_type == "l2" || basis_type == "DG" ||
          basis_type == "dg")
      {
         auto mu = options["space-dis"].value("sipg-penalty", -1.0);
         if (mu < 0)
         {
            auto degree = options["space-dis"]["degree"].get<double>();
            mu = pow(degree + 1, 3);
         }
         res.addInteriorFaceIntegrator(
             new DGInteriorFaceDiffusionIntegrator(nu, mu));
         std::cout << "adding sipg integ!\n";
      }

      if (options.contains("bcs"))
      {
         const auto &bcs = options["bcs"];

         // dirichlet boundary condition
         if (bcs.contains("essential"))
         {
            if (basis_type == "L2" || basis_type == "l2" ||
                basis_type == "DG" || basis_type == "dg")
            {
               std::vector<int> bdr_attr_marker;
               if (bcs["essential"].is_array())
               {
                  bdr_attr_marker = bcs["essential"].get<std::vector<int>>();
               }
               else
               {
                  throw MachException(
                      "Unrecognized JSON type for boundary attrs!");
               }

               auto mu = options["space-dis"].value("sipg-penalty", -1.0);
               if (mu < 0)
               {
                  auto degree = options["space-dis"]["degree"].get<double>();
                  mu = pow(degree + 1, 3);
               }
               std::cout << "mu: " << mu << "\n";
               res.addBdrFaceIntegrator(
                   new NonlinearDGDiffusionIntegrator(nu, *g, mu),
                   bdr_attr_marker);
            }
         }

         if (bcs.contains("weak-essential"))
         {
            std::vector<int> bdr_attr_marker;
            if (bcs["weak-essential"].is_array())
            {
               bdr_attr_marker = bcs["weak-essential"].get<std::vector<int>>();
            }
            else
            {
               throw MachException(
                   "Unrecognized JSON type for boundary attrs!");
            }

            auto mu = options["space-dis"].value("sipg-penalty", -1.0);
            if (mu < 0)
            {
               auto degree = options["space-dis"]["degree"].get<double>();
               mu = pow(degree + 1, 3);
            }
            std::cout << "mu: " << mu << "\n";
            res.addBdrFaceIntegrator(
                new NonlinearDGDiffusionIntegrator(nu, *g, mu),
                bdr_attr_marker);
         }
      }

      MachLinearForm linear_form(fes, fields);
      if (options.contains("current"))
      {
         current_coeff = std::make_unique<CurrentDensityCoefficient2D>(
             diff_stack, options["current"]);

         // MachInputs current_inputs = {{"current_density:phaseA", 0.0},
         //                              {"current_density:phaseB", 1.0},
         //                              {"current_density:phaseC", -1.0}};
         // setInputs(*current_coeff, current_inputs);

         // mfem::ParGridFunction j(&fes);
         // j.ProjectCoefficient(*current_coeff);
         // mfem::ParaViewDataCollection pv("CurrentDensity", fes.GetParMesh());
         // pv.SetPrefixPath("ParaView");
         // pv.SetLevelsOfDetail(3);
         // pv.SetDataFormat(mfem::VTKFormat::BINARY);
         // pv.SetHighOrderOutput(true);
         // pv.RegisterField("CurrentDensity", &j);
         // pv.Save();

         auto current_attrs = getCurrentAttributes(options);
         linear_form.addDomainIntegrator(
             new mach::DomainLFIntegrator(*current_coeff), current_attrs);
      }
      if (options.contains("magnets"))
      {
         mag_coeff = std::make_unique<MagnetizationCoefficient>(
             diff_stack, options["magnets"], materials, 2);

         nuM = std::make_unique<mach::ScalarVectorProductCoefficient>(
             nu, *mag_coeff);

         const auto &temp_field_iter = fields.find("temperature");

         auto &temp_field = temp_field_iter->second;

         auto magnet_attrs = getMagnetAttributes(options);
         linear_form.addDomainIntegrator(new MagnetizationSource2DIntegrator(
                                             *nuM, temp_field.gridFunc(), 1.0),
                                         magnet_attrs);
      }

      load = std::make_unique<MachLoad>(std::move(linear_form));
   }
   else
   {
      throw MachException(
          "Invalid mesh dimension for Magnetostatic Residual!\n");
   }
   setOptions(*this, options);
}

}  // namespace mach
