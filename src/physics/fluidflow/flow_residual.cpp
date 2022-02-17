#include "mfem.hpp"

#include "flow_residual.hpp"
#include "euler_fluxes.hpp"
#include "euler_integ.hpp"

using namespace std;
using namespace mfem;

namespace mach
{
FlowResidual::FlowResidual(const nlohmann::json &options,
                           ParFiniteElementSpace &fespace,
                           adept::Stack &diff_stack,
                           std::map<std::string, FiniteElementState> &fields)
 : fes(fespace),
   stack(diff_stack),
   // fields(std::make_unique<
   //        std::unordered_map<std::string, mfem::ParGridFunction>>()),
   res(fes, fields),
   ent(fes, fields),
   work(getSize(res))
{
   setOptions(*this, options);
   int dim = fes.GetMesh()->SpaceDimension();
   switch (dim)
   {
   case 1:
      addFlowIntegrators<1>(options);
      break;
   case 2:
      addFlowIntegrators<2>(options);
      break;
   case 3:
      addFlowIntegrators<3>(options);
      break;
   default:
      throw MachException("Invalid space dimension for FlowResidual!\n");
   }
}

template <int dim>
void FlowResidual::addFlowIntegrators(const nlohmann::json &options)
{
   if (!options.contains("flow-param") || !options.contains("space-dis") ||
       !options.contains("bcs"))
   {
      throw MachException(
          "FlowResidual::addFlowIntegrators: options must"
          "contain flow-param, space-dis, and bcs!\n");
   }
   nlohmann::json flow = options["flow-param"];
   nlohmann::json space_dis = options["space-dis"];
   nlohmann::json bcs = options["bcs"];
   if (state_is_entvar)
   {
      // Use entropy variables for the state
      addFlowDomainIntegrators<dim, true>(flow, space_dis);
      addFlowInterfaceIntegrators<dim, true>(flow, space_dis);
      addFlowBoundaryIntegrators<dim, true>(flow, space_dis, bcs);
      addEntropyIntegrators<dim, true>();
   }
   else
   {
      // Use the conservative variables for the state
      addFlowDomainIntegrators<dim>(flow, space_dis);
      addFlowInterfaceIntegrators<dim>(flow, space_dis);
      addFlowBoundaryIntegrators<dim>(flow, space_dis, bcs);
      addEntropyIntegrators<dim>();
   }
}

template <int dim, bool entvar>
void FlowResidual::addFlowDomainIntegrators(const nlohmann::json &flow,
                                            const nlohmann::json &space_dis)
{
   auto flux = space_dis.value("flux-fun", "Euler");
   if (flux == "IR")
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
void FlowResidual::addFlowInterfaceIntegrators(const nlohmann::json &flow,
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
void FlowResidual::addFlowBoundaryIntegrators(const nlohmann::json &flow,
                                              const nlohmann::json &space_dis,
                                              const nlohmann::json &bcs)
{
   if (bcs.contains("vortex"))
   {  // isentropic vortex BC
      if (dim != 2)
      {
         throw MachException(
             "FlowResidual::addFlowBoundaryIntegrators()\n"
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
void FlowResidual::addEntropyIntegrators()
{
   ent.addOutputDomainIntegrator(new EntropyIntegrator<dim, entvar>(stack));
}

int getSize(const FlowResidual &residual) { return getSize(residual.res); }

void setInputs(FlowResidual &residual, const MachInputs &inputs)
{
   // What if aoa_fs or mach_fs are being changed?
   setInputs(residual.res, inputs);
}

void setOptions(FlowResidual &residual, const nlohmann::json &options)
{
   residual.is_implicit = options.value("implicit", false);
   // define free-stream parameters; may or may not be used, depending on case
   residual.mach_fs = options["flow-param"]["mach"].get<double>();
   residual.aoa_fs = options["flow-param"].value("aoa", 0.0) * M_PI / 180;
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

void evaluate(FlowResidual &residual, const MachInputs &inputs, Vector &res_vec)
{
   evaluate(residual.res, inputs, res_vec);
}

mfem::Operator &getJacobian(FlowResidual &residual,
                            const MachInputs &inputs,
                            const string &wrt)
{
   return getJacobian(residual.res, inputs, wrt);
}

double calcEntropy(FlowResidual &residual, const MachInputs &inputs)
{
   return calcOutput(residual.ent, inputs);
}

double calcEntropyChange(FlowResidual &residual, const MachInputs &inputs)
{
   Vector x;
   setVectorFromInputs(inputs, "state", x, false, true);
   Vector dxdt;
   setVectorFromInputs(inputs, "state_dot", dxdt, false, true);
   double dt = NAN;
   double time = NAN;
   setValueFromInputs(inputs, "time", time, true);
   setValueFromInputs(inputs, "dt", dt, true);
   auto &y = residual.work;
   add(x, dt, dxdt, y);
   auto form_inputs = MachInputs({{"state", y}, {"time", time + dt}});
   return calcFormOutput(residual.res, form_inputs);
}

}  // namespace mach
