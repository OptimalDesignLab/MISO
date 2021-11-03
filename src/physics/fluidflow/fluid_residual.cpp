#include "mfem.hpp"

#include "fluid_residual.hpp"
#include "euler_fluxes.hpp"
#include "euler_integ.hpp"

using namespace std;
using namespace mfem;

namespace mach
{
FluidResidual::FluidResidual(const nlohmann::json &options,
                             ParFiniteElementSpace &fespace,
                             adept::Stack &diff_stack)
 : fes(fespace),
   stack(diff_stack),
   fields(std::make_unique<
          std::unordered_map<std::string, mfem::ParGridFunction>>()),
   res(fes, *fields),
   ent(fes, *fields)
{
   setOptions(*this, options);
   int dim = fes.GetMesh()->SpaceDimension();
   switch (dim)
   {
   case 1:
      addFluidIntegrators<1>(options);
      break;
   case 2:
      addFluidIntegrators<2>(options);
      break;
   case 3:
      addFluidIntegrators<3>(options);
      break;
   default:
      throw MachException("Invalid space dimension for FluidResidual!\n");
   }
}

template <int dim>
void FluidResidual::addFluidIntegrators(const nlohmann::json &options)
{
   if (!options.contains("flow-param") || !options.contains("space-dis") ||
       !options.contains("bcs"))
   {
      throw MachException(
          "FluidResidual::addFluidIntegrators: options must"
          "contain flow-param, space-dis, and bcs!\n");
   }
   nlohmann::json flow = options["flow-param"];
   nlohmann::json space_dis = options["space-dis"];
   nlohmann::json bcs = options["bcs"];
   if (state_is_entvar)
   {
      // Use entropy variables for the state
      addFluidDomainIntegrators<dim, true>(flow, space_dis);
      addFluidInterfaceIntegrators<dim, true>(flow, space_dis);
      addFluidBoundaryIntegrators<dim, true>(flow, space_dis, bcs);
      addEntropyIntegrators<dim, true>();
   }
   else
   {
      // Use the conservative variables for the state
      addFluidDomainIntegrators<dim>(flow, space_dis);
      addFluidInterfaceIntegrators<dim>(flow, space_dis);
      addFluidBoundaryIntegrators<dim>(flow, space_dis, bcs);
      addEntropyIntegrators<dim>();
   }
}

template <int dim, bool entvar>
void FluidResidual::addFluidDomainIntegrators(const nlohmann::json &flow,
                                              const nlohmann::json &space_dis)
{
   if (space_dis["flux-fun"].get<string>() == "IR")
   {
      res.addDomainIntegrator(new IsmailRoeIntegrator<dim, entvar>(stack));
   }
   else
   {
      if (entvar)
      {
         throw MachException(
             "Invalid inviscid integrator for entropy"
             " state!\n");
      }
      res.addDomainIntegrator(new EulerIntegrator<dim>(stack));
   }
   // add the LPS stabilization, if necessary
   auto lps_coeff = space_dis.value("lps-coeff", 0.0);
   if (lps_coeff > 0.0)
   {
      res.addDomainIntegrator(
          new EntStableLPSIntegrator<dim, entvar>(stack, lps_coeff));
   }
}

template <int dim, bool entvar>
void FluidResidual::addFluidInterfaceIntegrators(
    const nlohmann::json &flow,
    const nlohmann::json &space_dis)
{
   // add the integrators based on if discretization is continuous or discrete
   if (space_dis["basis-type"].get<string>() == "dsbp")
   {
      auto iface_coeff = space_dis.value("iface-coeff", 0.0);
      res.addInteriorFaceIntegrator(new InterfaceIntegrator<dim, entvar>(
          stack, iface_coeff, fes.FEColl()));
   }
}

template <int dim, bool entvar>
void FluidResidual::addFluidBoundaryIntegrators(const nlohmann::json &flow,
                                                const nlohmann::json &space_dis,
                                                const nlohmann::json &bcs)
{
   if (bcs.contains("vortex"))
   {  // isentropic vortex BC
      if (dim != 2)
      {
         throw MachException(
             "FluidResidual::addFluidBoundaryIntegrators()\n"
             "\tisentropic vortex BC must use 2D mesh!");
      }
      vector<int> bdr_attr_marker = bcs["vortex"].get<vector<int>>();
      res.addBdrFaceIntegrator(
          new IsentropicVortexBC<dim, entvar>(stack, fes.FEColl()),
          bdr_attr_marker);
   }
   if (bcs.contains("slip-wall"))
   {  // slip-wall boundary condition
      vector<int> bdr_attr_marker = bcs["slip-wall"].get<vector<int>>();
      res.addBdrFaceIntegrator(new SlipWallBC<dim, entvar>(stack, fes.FEColl()),
                               bdr_attr_marker);
   }
   if (bcs.contains("far-field"))
   {
      // far-field boundary conditions
      vector<int> bdr_attr_marker = bcs["far-field"].get<vector<int>>();
      mfem::Vector qfar(dim + 2);
      getFreeStreamQ<double, dim, entvar>(
          mach_fs, aoa_fs, iroll, ipitch, qfar.GetData());
      res.addBdrFaceIntegrator(
          new FarFieldBC<dim, entvar>(stack, fes.FEColl(), qfar),
          bdr_attr_marker);
   }
}

template <int dim, bool entvar>
void FluidResidual::addEntropyIntegrators()
{
   ent.addOutputDomainIntegrator(new EntropyIntegrator<dim, entvar>(stack));
}

int getSize(const FluidResidual &residual) { return getSize(residual.res); }

void setInputs(FluidResidual &residual, const MachInputs &inputs)
{
   // What if aoa_fs or mach_fs are being changed?
   setInputs(residual.res, inputs);
}

void setOptions(FluidResidual &residual, const nlohmann::json &options)
{
   residual.is_implicit = options.value("implicit", false);
   // define free-stream parameters; may or may not be used, depending on case
   residual.mach_fs = options["flow-param"]["mach"].get<double>();
   residual.aoa_fs = options["flow-param"]["aoa"].get<double>() * M_PI / 180;
   residual.iroll = options["flow-param"].value("roll-axis", 0);
   residual.ipitch = options["flow-param"].value("pitch-axis", 1);
   residual.state_is_entvar = options["flow-param"].value("entvar", false);
   if (residual.iroll == residual.ipitch)
   {
      throw MachException("iroll and ipitch must be distinct dimensions!");
   }
   if ((residual.iroll < 0) || (residual.iroll > 2))
   {
      throw MachException("iroll axis must be between 0 and 2!");
   }
   if ((residual.ipitch < 0) || (residual.ipitch > 2))
   {
      throw MachException("ipitch axis must be between 0 and 2!");
   }
   setOptions(residual.res, options);
}

void evaluate(FluidResidual &residual,
              const MachInputs &inputs,
              Vector &res_vec)
{
   evaluate(residual.res, inputs, res_vec);
}

mfem::Operator &getJacobian(FluidResidual &residual,
                            const MachInputs &inputs,
                            string wrt)
{
   return getJacobian(residual.res, inputs, std::move(wrt));
}

double calcEntropy(FluidResidual &residual, const MachInputs &inputs)
{
   return calcOutput(residual.ent, inputs);
}

}  // namespace mach
