#include "mfem.hpp"

#include "flow_residual.hpp"
#include "euler_fluxes.hpp"
#include "euler_integ.hpp"

using namespace std;
using namespace mfem;

namespace mach
{
template <int dim, bool entvar>
FlowResidual<dim, entvar>::FlowResidual(const nlohmann::json &options,
                                        ParFiniteElementSpace &fespace,
                                        adept::Stack &diff_stack)
 : fes(fespace),
   stack(diff_stack),
   fields(std::make_unique<
          std::unordered_map<std::string, mfem::ParGridFunction>>()),
   res(fes, *fields),
   ent(fes, *fields),
   work(getSize(res))
{
   setOptions_(options);
   addFlowIntegrators(options);
}

template <int dim, bool entvar>
void FlowResidual<dim, entvar>::addFlowIntegrators(
   const nlohmann::json &options)
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
   addFlowDomainIntegrators(flow, space_dis);
   addFlowInterfaceIntegrators(flow, space_dis);
   addFlowBoundaryIntegrators(flow, space_dis, bcs);
   addEntropyIntegrators();
}

template <int dim, bool entvar>
void FlowResidual<dim, entvar>::addFlowDomainIntegrators(
   const nlohmann::json &flow,
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
void FlowResidual<dim, entvar>::addFlowInterfaceIntegrators(
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
void FlowResidual<dim, entvar>::addFlowBoundaryIntegrators(
   const nlohmann::json &flow, const nlohmann::json &space_dis,
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
void FlowResidual<dim, entvar>::addEntropyIntegrators()
{
   ent.addOutputDomainIntegrator(new EntropyIntegrator<dim, entvar>(stack));
}

template <int dim, bool entvar>
int FlowResidual<dim, entvar>::getSize_() const
{
   return getSize(res);
}

template <int dim, bool entvar>
void FlowResidual<dim, entvar>::setInputs_(const MachInputs &inputs)
{
   // What if aoa_fs or mach_fs are being changed?
   setInputs(res, inputs);
}

template <int dim, bool entvar>
void FlowResidual<dim, entvar>::setOptions_(const nlohmann::json &options)
{
   is_implicit = options.value("implicit", false);
   // define free-stream parameters; may or may not be used, depending on case
   mach_fs = options["flow-param"]["mach"].get<double>();
   aoa_fs = options["flow-param"].value("aoa", 0.0) * M_PI / 180;
   iroll = options["flow-param"].value("roll-axis", 0);
   ipitch = options["flow-param"].value("pitch-axis", 1);
   state_is_entvar = options["flow-param"].value("entropy-state", false);
   if (iroll == ipitch)
   {
      throw MachException("iroll and ipitch must be distinct dimensions!");
   }
   if ((iroll < 0) || (iroll > 2))
   {
      throw MachException("iroll axis must be between 0 and 2!");
   }
   if ((ipitch < 0) || (ipitch > 2))
   {
      throw MachException("ipitch axis must be between 0 and 2!");
   }
   setOptions(res, options);
}

template <int dim, bool entvar>
void FlowResidual<dim, entvar>::evaluate_(const MachInputs &inputs,
                                         Vector &res_vec)
{
   evaluate(res, inputs, res_vec);
}

template <int dim, bool entvar>
mfem::Operator &FlowResidual<dim, entvar>::getJacobian_(
   const MachInputs &inputs, const string &wrt)
{
   return getJacobian(res, inputs, wrt);
}

template <int dim, bool entvar>
double FlowResidual<dim, entvar>::calcEntropy_(const MachInputs &inputs)
{
   return calcOutput(ent, inputs);
}

template <int dim, bool entvar>
double FlowResidual<dim, entvar>::calcEntropyChange_(const MachInputs &inputs)
{
   Vector x;
   setVectorFromInputs(inputs, "state", x, false, true);
   Vector dxdt;
   setVectorFromInputs(inputs, "state_dot", dxdt, false, true);
   double dt = NAN;
   double time = NAN;
   setValueFromInputs(inputs, "time", time, true);
   setValueFromInputs(inputs, "dt", dt, true);
   auto &y = work;
   add(x, dt, dxdt, y);
   auto form_inputs = MachInputs({{"state", y}, {"time", time + dt}});
   return calcFormOutput(res, form_inputs);
}

template <int dim, bool entvar>
double FlowResidual<dim, entvar>::minCFLTimeStep(
    double cfl,
    const mfem::ParGridFunction &state)
{
   double dt_local = 1e100;
   Vector xi(dim);
   Vector dxij(dim);
   Vector ui;
   Vector dxidx;
   DenseMatrix uk;
   DenseMatrix adjJt(dim);
   for (int k = 0; k < fes.GetNE(); k++)
   {
      // get the element, its transformation, and the state values on element
      const FiniteElement *fe = fes.GetFE(k);
      const IntegrationRule *ir = &(fe->GetNodes());
      ElementTransformation *trans = fes.GetElementTransformation(k);
      state.GetVectorValues(*trans, *ir, uk);
      for (int i = 0; i < fe->GetDof(); ++i)
      {
         trans->SetIntPoint(&fe->GetNodes().IntPoint(i));
         trans->Transform(fe->GetNodes().IntPoint(i), xi);
         CalcAdjugateTranspose(trans->Jacobian(), adjJt);
         uk.GetColumnReference(i, ui);
         for (int j = 0; j < fe->GetDof(); ++j)
         {
            if (j == i)
            {
               continue;
            }
            trans->Transform(fe->GetNodes().IntPoint(j), dxij);
            dxij -= xi;
            double dx = dxij.Norml2();
            dt_local =
                min(dt_local,
                    cfl * dx * dx /
                     calcSpectralRadius<double, dim, entvar>(dxij, ui)); // extra dx is to normalize dxij
         }
      }
   }
   double dt_min = NAN;
   MPI_Allreduce(&dt_local,
                 &dt_min,
                 1,
                 MPI_DOUBLE,
                 MPI_MIN,
                 state.ParFESpace()->GetComm());
   return dt_min;
}

template <int dim, bool entvar>
double FlowResidual<dim, entvar>::calcConservativeVarsL2Error(
    const mfem::ParGridFunction &state,
    void (*u_exact)(const mfem::Vector &, mfem::Vector &),
    int entry)
{
   // Following function, defined earlier in file, computes the error at a node
   // Beware: this is not particularly efficient, and **NOT thread safe!**
   Vector qdiscrete(dim + 2);
   Vector qexact(dim + 2);  // define here to avoid reallocation
   auto node_error = [&](const Vector &discrete, const Vector &exact) -> double
   {
      if constexpr(entvar)
      {
         calcConservativeVars<double, dim>(discrete.GetData(),
                                           qdiscrete.GetData());
         calcConservativeVars<double, dim>(exact.GetData(), qexact.GetData());
      }
      else
      {
         qdiscrete = discrete;
         qexact = exact;
      }
      double err = 0.0;
      if (entry < 0)
      {
         for (int i = 0; i < dim + 2; ++i)
         {
            double dq = qdiscrete(i) - qexact(i);
            err += dq * dq;
         }
      }
      else
      {
         err = qdiscrete(entry) - qexact(entry);
         err = err * err;
      }
      return err;
   };
   VectorFunctionCoefficient exsol(dim + 2, u_exact);
   DenseMatrix vals;
   DenseMatrix exact_vals;
   Vector u_j;
   Vector exsol_j;
   double loc_norm = 0.0;
   for (int i = 0; i < fes.GetNE(); i++)
   {
      const FiniteElement *fe = fes.GetFE(i);
      const IntegrationRule *ir = &(fe->GetNodes());
      ElementTransformation *T = fes.GetElementTransformation(i);
      state.GetVectorValues(*T, *ir, vals);
      exsol.Eval(exact_vals, *T, *ir);
      for (int j = 0; j < ir->GetNPoints(); j++)
      {
         const IntegrationPoint &ip = ir->IntPoint(j);
         T->SetIntPoint(&ip);
         vals.GetColumnReference(j, u_j);
         exact_vals.GetColumnReference(j, exsol_j);
         loc_norm += ip.weight * T->Weight() * node_error(u_j, exsol_j);
      }
   }
   double norm = NAN;
   MPI_Allreduce(
       &loc_norm, &norm, 1, MPI_DOUBLE, MPI_SUM, state.ParFESpace()->GetComm());
   if (norm < 0.0)  // This was copied from mfem...should not happen for us
   {
      return -sqrt(-norm);
   }
   return sqrt(norm);
}

// explicit instantiation
template class FlowResidual<1, true>;
template class FlowResidual<1, false>;
template class FlowResidual<2, true>;
template class FlowResidual<2, false>;
template class FlowResidual<3, true>;
template class FlowResidual<3, false>;

}  // namespace mach
