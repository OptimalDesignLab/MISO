#include <iostream>
#include <memory>
#include <string>

#include "mfem.hpp"
#include "nlohmann/json.hpp"

#include "electromag_integ.hpp"
#include "coefficient.hpp"
#include "miso_input.hpp"
#include "thermal_integ.hpp"

#include "thermal_residual.hpp"
#include "utils.hpp"

namespace miso
{
int getSize(const ThermalResidual &residual) { return getSize(residual.res); }

void setInputs(ThermalResidual &residual, const miso::MISOInputs &inputs)
{
   residual.kappa->setInputs(inputs);
   residual.rho->setInputs(inputs);

   setInputs(residual.res, inputs);

   setVectorFromInputs(inputs, "thermal_load", residual.load);
   if (residual.load.Size() != 0)
   {
      // std::cout << "residual.load.Size() != 0 and is:\n";
      // residual.load.Print(mfem::out, 25);
      // std::cout << "That has been the thermal load vector\n";

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

void finalizeAdjointSystem(ThermalResidual &residual,
                           mfem::Solver &adj_solver,
                           const MISOInputs &inputs,
                           mfem::Vector &state_bar,
                           mfem::Vector &adjoint)
{
   finalizeAdjointSystem(residual.res, adj_solver, inputs, state_bar, adjoint);
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
   // if wrt starts with prefix "thermal_load"
   if (wrt.rfind("thermal_load", 0) == 0)
   {
      /// TODO: Test this implementation
      // Thermal load is subtracted from the (stiffness matrix * thermal) state
      // term Therefore derivative of residual w/r/t thermal_load is negative
      // identity matrix (represented as negative of input vector in matrix-free
      // form)
      res_dot = 0.0;
      res_dot.Add(-1.0, wrt_dot);
      return;
   }
   // if wrt starts with prefix "temperature"
   /// TODO: Determine if this should be "state" instead of temperature
   else if (wrt.rfind("temperature", 0) == 0)
   {
      /// NOTE: Derivative of the thermal residual wrt the thermal state is
      /// Jacobian (already implemented above)
      return;
   }
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
   // if wrt starts with prefix "thermal_load"
   if (wrt.rfind("thermal_load", 0) == 0)
   {
      /// TODO: Test this implementation
      // Thermal load is subtracted from the (stiffness matrix * thermal) state
      // term Therefore derivative of residual w/r/t thermal_load is negative
      // identity matrix (represented as negative of input vector in matrix-free
      // form)
      wrt_bar = 0.0;
      wrt_bar.Add(-1.0, res_bar);
      return;
   }
   // if wrt starts with prefix "temperature"
   /// TODO: Determine if this should be "state" instead of temperature
   else if (wrt.rfind("temperature", 0) == 0)
   {
      /// NOTE: Derivative of the thermal residual wrt the thermal state is
      /// Jacobian (already implemented above)
      return;
   }
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
   g(std::make_unique<MeshDependentCoefficient>(
       std::make_unique<mfem::GridFunctionCoefficient>(
           &fields.at("dirichlet_bc").gridFunc()))),
   kappa(
       constructMaterialCoefficient("kappa", options["components"], materials)),
   rho(constructMaterialCoefficient("rho", options["components"], materials)),
   // cv(constructMaterialCoefficient("cv", options["components"], materials)),
   prec(constructPreconditioner(fes, options["lin-prec"]))

{
   res.addDomainIntegrator(new NonlinearDiffusionIntegrator(*kappa));

   const auto &basis_type =
       options["space-dis"]["basis-type"].get<std::string>();

   if (basis_type == "L2" || basis_type == "l2" || basis_type == "DG" ||
       basis_type == "dg")
   {
      auto mu = options["space-dis"].value("sipg-penalty", -1.0);
      if (mu < 0)
      {
         auto degree = options["space-dis"]["degree"].get<double>();
         mu = pow(degree + 1, 2);
      }
      res.addInteriorFaceIntegrator(
          new DGInteriorFaceDiffusionIntegrator(*kappa, mu));
      std::cout << "adding sipg integ!\n";
   }

   if (options.contains("bcs"))
   {
      const auto &bcs = options["bcs"];

      // convection heat transfer boundary condition
      if (bcs.contains("convection"))
      {
         auto &bc = bcs["convection"];
         std::vector<int> bdr_attr_marker;
         double h = 1.0;
         double theta_f = 1.0;
         if (bc.is_array())
         {
            bdr_attr_marker = bc.get<std::vector<int>>();
         }
         else if (bc.is_object())
         {
            bdr_attr_marker = bc["attrs"].get<std::vector<int>>();
            h = bc.value("h", h);
            theta_f = bc.value("fluid_temp", theta_f);
         }

         res.addBdrFaceIntegrator(new ConvectionBCIntegrator(h, theta_f),
                                  bdr_attr_marker);
      }

      if (bcs.contains("outflux"))
      {
         auto &bc = bcs["outflux"];
         std::vector<int> bdr_attr_marker;
         double flux = 1.0;
         if (bc.is_array())
         {
            bdr_attr_marker = bc.get<std::vector<int>>();
         }
         else if (bc.is_object())
         {
            bdr_attr_marker = bc["attrs"].get<std::vector<int>>();
            flux = bc.value("flux", flux);
         }

         res.addBdrFaceIntegrator(new OutfluxBCIntegrator(flux, 1.0),
                                  bdr_attr_marker);
      }

      // dirichlet boundary condition
      if (bcs.contains("essential"))
      {
         if (basis_type == "L2" || basis_type == "l2" || basis_type == "DG" ||
             basis_type == "dg")
         {
            std::vector<int> bdr_attr_marker;
            if (bcs["essential"].is_array())
            {
               bdr_attr_marker = bcs["essential"].get<std::vector<int>>();
            }
            else
            {
               throw MISOException(
                   "Unrecognized JSON type for boundary attrs!");
            }

            auto mu = options["space-dis"].value("sipg-penalty", -1.0);
            if (mu < 0)
            {
               auto degree = options["space-dis"]["degree"].get<double>();
               mu = pow(degree + 1, 2);
            }
            std::cout << "mu: " << mu << "\n";
            res.addBdrFaceIntegrator(
                new NonlinearDGDiffusionIntegrator(*kappa, *g, mu),
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
            throw MISOException("Unrecognized JSON type for boundary attrs!");
         }

         auto mu = options["space-dis"].value("sipg-penalty", -1.0);
         if (mu < 0)
         {
            auto degree = options["space-dis"]["degree"].get<double>();
            mu = pow(degree + 1, 2);
         }
         std::cout << "mu: " << mu << "\n";
         res.addBdrFaceIntegrator(
             new NonlinearDGDiffusionIntegrator(*kappa, *g, mu),
             bdr_attr_marker);
      }
   }

   if (options.contains("interfaces"))
   {
      const auto &interfaces = options["interfaces"];

      for (const auto &[kind, interface] : interfaces.items())
      {
         std::cout << "kind: " << kind << "\n";
         std::cout << "interface: " << interface << "\n";
         // thermal contact resistance interface
         if (kind == "thermal_contact_resistance")
         {
            for (const auto &[name, intr] : interface.items())
            {
               const auto &attrs = intr["attrs"].get<std::vector<int>>();

               auto mu = options["space-dis"].value("sipg-penalty", -1.0);
               if (mu < 0)
               {
                  auto degree = options["space-dis"]["degree"].get<double>();
                  mu = pow(degree + 1, 2);
               }

               res.addInternalBoundaryFaceIntegrator(
                   new DGInteriorFaceDiffusionIntegrator(*kappa, mu, -1),
                   attrs);

               const auto h = intr["h"].get<double>();

               auto *integ = new ThermalContactResistanceIntegrator(h, name);
               // setInputs(*integ, {{"h", h}});
               res.addInternalBoundaryFaceIntegrator(integ, attrs);

               std::cout << "adding " << name << " TCR integrator!\n";
               std::cout << "with attrs: " << intr["attrs"] << "\n";
            }
         }
         else if (kind == "convection")
         {
            for (const auto &[name, intr] : interface.items())
            {
               const auto &attrs = intr["attrs"].get<std::vector<int>>();

               auto mu = options["space-dis"].value("sipg-penalty", -1.0);
               if (mu < 0)
               {
                  auto degree = options["space-dis"]["degree"].get<double>();
                  mu = pow(degree + 1, 2);
               }

               res.addInternalBoundaryFaceIntegrator(
                   new DGInteriorFaceDiffusionIntegrator(*kappa, mu, -1),
                   attrs);

               const auto h = intr["h"].get<double>();
               const auto theta_f = intr["fluid_temp"].get<double>();

               auto *integ =
                   new InternalConvectionInterfaceIntegrator(h, theta_f, name);
               res.addInternalBoundaryFaceIntegrator(integ, attrs);

               // auto *integ = new ThermalContactResistanceIntegrator(h,
               // name);
               // // setInputs(*integ, {{"h", h}});
               // res.addInternalBoundaryFaceIntegrator(integ, attrs);

               std::cout << "adding " << name
                         << " internal convection integrator!\n";
               std::cout << "with attrs: " << intr["attrs"] << "\n";
            }
         }
      }
   }
}

}  // namespace miso
