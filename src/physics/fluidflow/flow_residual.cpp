#include <memory>

#include "mfem.hpp"

#include "diag_mass_integ.hpp"
#include "flow_residual.hpp"
#include "euler_fluxes.hpp"
#include "euler_integ.hpp"
#include "navier_stokes_integ.hpp"
#include "utils.hpp"

using namespace std;
using namespace mfem;

namespace
{
/// Constructs a preconditioner for the flow residual's state Jacobian
/// \param[in] prec_options - options specific to the preconditioner
/// \return the constructed preconditioner, and the returned pointer is owned
/// by the `residual`
std::unique_ptr<mfem::Solver> constructPreconditioner(
    mfem::ParFiniteElementSpace &fes,
    const nlohmann::json &prec_options)
{
   std::unique_ptr<mfem::Solver> prec;

   auto prec_type = prec_options["type"].get<std::string>();
   if (prec_type == "hypreeuclid")
   {
      prec = std::make_unique<HypreEuclid>(fes.GetComm());
      // TODO: need to add HYPRE_EuclidSetLevel to odl branch of mfem
      out << "WARNING! Euclid fill level is hard-coded"
          << "(see AbstractSolver::constructLinearSolver() for details)"
          << endl;
      // int fill = options["lin-solver"]["filllevel"].get<int>();
      // HYPRE_EuclidSetLevel(dynamic_cast<HypreEuclid*>(precond.get())->GetPrec(),
      // fill);
   }
   else if (prec_type == "hypreilu")
   {
      prec = std::make_unique<HypreILU>();
      auto *ilu = dynamic_cast<HypreILU *>(prec.get());
      HYPRE_ILUSetType(*ilu, prec_options["ilu-type"]);
      HYPRE_ILUSetLevelOfFill(*ilu, prec_options["lev-fill"]);
      HYPRE_ILUSetLocalReordering(*ilu, prec_options["ilu-reorder"]);
      HYPRE_ILUSetPrintLevel(*ilu, prec_options["printlevel"]);
      // Just listing the options below in case we need them in the future
      // HYPRE_ILUSetSchurMaxIter(ilu, schur_max_iter);
      // HYPRE_ILUSetNSHDropThreshold(ilu, nsh_thres); needs type = 20,21
      // HYPRE_ILUSetDropThreshold(ilu, drop_thres);
      // HYPRE_ILUSetMaxNnzPerRow(ilu, nz_max);
   }
   else if (prec_type == "hypreams")
   {
      prec = std::make_unique<HypreAMS>(&fes);
      auto *ams = dynamic_cast<HypreAMS *>(prec.get());
      ams->SetPrintLevel(prec_options["printlevel"]);
      ams->SetSingularProblem();
   }
   else if (prec_type == "hypreboomeramg")
   {
      prec = std::make_unique<HypreBoomerAMG>();
      auto *amg = dynamic_cast<HypreBoomerAMG *>(prec.get());
      amg->SetPrintLevel(prec_options["printlevel"]);
   }
   else if (prec_type == "blockilu")
   {
      prec = std::make_unique<BlockILU>(fes.GetVDim());
   }
   else
   {
      throw mach::MachException(
          "Unsupported preconditioner type!\n"
          "\tavilable options are: HypreEuclid, HypreILU, HypreAMS,"
          " HypreBoomerAMG.\n");
   }
   return prec;
}

}  // namespace

namespace mach
{
template <int dim, bool entvar>
FlowResidual<dim, entvar>::FlowResidual(
    const nlohmann::json &options,
    ParFiniteElementSpace &fespace,
    std::map<std::string, FiniteElementState> &fields,
    adept::Stack &diff_stack,
    ostream &outstream)
 : options(options),
   out(outstream),
   fes(fespace),
   stack(diff_stack),
   fields(fields),
   res(fes, fields),
   mass(&fespace),
   ent(fes, fields),
   work(getSize(res))
{
   setOptions_(options);

   if (!options.contains("flow-param") || !options.contains("space-dis"))
   {
      throw MachException(
          "FlowResidual::addFlowIntegrators: options must"
          "contain flow-param and space-dis!\n");
   }
   const nlohmann::json &flow = options["flow-param"];
   const nlohmann::json &space_dis = options["space-dis"];
   addFlowDomainIntegrators(flow, space_dis);
   addFlowInterfaceIntegrators(flow, space_dis);
   if (options.contains("bcs"))
   {
      const nlohmann::json &bcs = options["bcs"];
      addFlowBoundaryIntegrators(flow, space_dis, bcs);
   }

   // set up the mass bilinear form, but do not construct the matrix unless
   // necessary (see getMassMatrix_)
   const char *name = fes.FEColl()->Name();
   if ((strncmp(name, "SBP", 3) == 0) || (strncmp(name, "DSBP", 4) == 0))
   {
      bool space_vary = options["time-dis"]["steady"];
      mass.AddDomainIntegrator(
          new DiagMassIntegrator(fes.GetVDim(), space_vary));
   }
   else
   {
      mass.AddDomainIntegrator(new mfem::MassIntegrator());
   }

   // Set up the entropy function integrators
   addEntropyIntegrators();
}

template <int dim, bool entvar>
void FlowResidual<dim, entvar>::addFlowDomainIntegrators(
    const nlohmann::json &flow,
    const nlohmann::json &space_dis)
{
   const auto &flux = space_dis["flux-fun"];
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
   const auto &lps_coeff = space_dis["lps-coeff"];
   if (lps_coeff > 0.0)
   {
      res.addDomainIntegrator(
          new EntStableLPSIntegrator<dim, entvar>(stack, lps_coeff));
   }
   // add viscous volume integrators, if necessary
   if (flow["viscous"])
   {
      res.addDomainIntegrator(
          new ESViscousIntegrator<dim>(stack, re_fs, pr_fs, mu));
      if (flow["viscous-mms"])
      {
         if (dim != 2)
         {
            throw MachException("Viscous MMS problem only available for 2D!");
         }
         res.addDomainIntegrator(new NavierStokesMMSIntegrator(re_fs, pr_fs));
      }
   }
}

template <int dim, bool entvar>
void FlowResidual<dim, entvar>::addFlowInterfaceIntegrators(
    const nlohmann::json &flow,
    const nlohmann::json &space_dis)
{
   // add the integrators based on if discretization is continuous or discrete
   if (space_dis["basis-type"] == "dsbp")
   {
      const auto &iface_coeff = space_dis["iface-coeff"];
      res.addInteriorFaceIntegrator(new InterfaceIntegrator<dim, entvar>(
          stack, iface_coeff, fes.FEColl()));
      if (flow["viscous"])
      {
         throw MachException(
             "Viscous DG interface terms have not been"
             " implemented!");
      }
   }
}

template <int dim, bool entvar>
void FlowResidual<dim, entvar>::addFlowBoundaryIntegrators(
    const nlohmann::json &flow,
    const nlohmann::json &space_dis,
    const nlohmann::json &bcs)
{
   if (flow["viscous"])
   {
      addViscousBoundaryIntegrators(flow, space_dis, bcs);
   }
   else
   {
      addInviscidBoundaryIntegrators(flow, space_dis, bcs);
   }
}

template <int dim, bool entvar>
void FlowResidual<dim, entvar>::addInviscidBoundaryIntegrators(
    const nlohmann::json &flow,
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
      getFreeStreamState(qfar);
      res.addBdrFaceIntegrator(
          new FarFieldBC<dim, entvar>(stack, fes.FEColl(), qfar),
          bdr_attr_marker);
   }
   if (bcs.contains("pump"))
   {
      // Boundary is forced in an oscillatory way; the BC function should be
      // passed in
      vector<int> bdr_attr_marker = bcs["pump"].get<vector<int>>();
      // define the boundary condition state function
      auto pump = [](double t, const Vector &x, Vector &q)
      {
         double uL = 0.05;
         double xL = uL * (1.0 - cos(t));
         q[0] = 1.0 / (1.0 - xL);
         q[1] = q[0] * uL * sin(t);
         q[2] = 0.0;
         double press = pow(q[0], mach::euler::gamma);
         q[3] = press / mach::euler::gami + 0.5 * q[1] * q[1] / q[0];
      };
      res.addBdrFaceIntegrator(
          new EntropyConserveBC<dim, entvar>(stack, fes.FEColl(), pump),
          bdr_attr_marker);
   }
   if (bcs.contains("control"))
   {
      // Boundary control
      vector<int> bdr_attr_marker = bcs["control"].get<vector<int>>();
      // define the scaling function for the control
      std::function<double(double, const Vector &, const Vector &)> scale =
          squaredExponential;
      // Should pass in xc and len...
      double len = 0.1;
      Vector xc({0.5, 0.0});
      res.addBdrFaceIntegrator(
          new ControlBC<dim, entvar>(stack, fes.FEColl(), scale, xc, len),
          bdr_attr_marker);
   }
}

template <int dim, bool entvar>
void FlowResidual<dim, entvar>::addViscousBoundaryIntegrators(
    const nlohmann::json &flow,
    const nlohmann::json &space_dis,
    const nlohmann::json &bcs)
{
   if (bcs.contains("slip-wall"))
   {
      // slip-wall boundary condition with appropriate Neumann BCs
      vector<int> bdr_attr_marker = bcs["slip-wall"].get<vector<int>>();
      res.addBdrFaceIntegrator(
          new ViscousSlipWallBC<dim>(stack, fes.FEColl(), re_fs, pr_fs, mu),
          bdr_attr_marker);
   }
   if (bcs.contains("no-slip-adiabatic"))
   {
      // no-slip adiabatic wall BCs
      vector<int> bdr_attr_marker = bcs["no-slip-adiabatic"].get<vector<int>>();
      // reference state needed by penalty flux
      Vector q_ref(dim + 2);
      getFreeStreamState(q_ref);
      res.addBdrFaceIntegrator(
          new NoSlipAdiabaticWallBC<dim>(
              stack, fes.FEColl(), re_fs, pr_fs, q_ref, mu),
          bdr_attr_marker);
   }
   if (bcs.contains("far-field"))
   {
      // far-field boundary conditions
      vector<int> bdr_attr_marker = bcs["far-field"].get<vector<int>>();
      Vector qfar(dim + 2);
      getFreeStreamState(qfar);
      res.addBdrFaceIntegrator(
          new FarFieldBC<dim, entvar>(stack, fes.FEColl(), qfar),
          bdr_attr_marker);
   }
   if (bcs.contains("viscous-mms"))
   {
      // viscous MMS boundary conditions
      auto exactbc = [](const Vector &x, Vector &u)
      { viscousMMSExact<double>(x.GetData(), u.GetData()); };
      vector<int> bdr_attr_marker = bcs["viscous-mms"].get<vector<int>>();
      res.addBdrFaceIntegrator(
          new ViscousExactBC<dim>(
              stack, fes.FEColl(), re_fs, pr_fs, exactbc, mu),
          bdr_attr_marker);
   }
   if (bcs.contains("control"))
   {
      // Boundary control
      vector<int> bdr_attr_marker = bcs["control"].get<vector<int>>();
      // reference state needed by penalty flux
      Vector q_ref(dim + 2);
      getFreeStreamState(q_ref);
      // define the scaling function for the control
      std::function<double(double, const Vector &, const Vector &)> scale =
          squaredExponential;
      // Should pass in xc and len...
      double len = 0.1;
      Vector xc({0.5, 0.0});
      res.addBdrFaceIntegrator(
          new ViscousControlBC<dim>(
              stack, fes.FEColl(), re_fs, pr_fs, q_ref, scale, xc, len, mu),
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
   viscous = options["flow-param"]["viscous"];
   mu = options["flow-param"]["mu"];
   mach_fs = options["flow-param"]["mach"];
   aoa_fs = options["flow-param"]["aoa"].get<double>() * M_PI / 180;
   re_fs = options["flow-param"]["Re"];
   pr_fs = options["flow-param"]["Pr"];
   iroll = options["flow-param"]["roll-axis"];
   ipitch = options["flow-param"]["pitch-axis"];
   state_is_entvar = options["flow-param"]["entropy-state"];
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
   setInputs(res, inputs);
   evaluate(res, inputs, res_vec);
}

template <int dim, bool entvar>
mfem::Operator &FlowResidual<dim, entvar>::getJacobian_(
    const MachInputs &inputs,
    const string &wrt)
{
   return getJacobian(res, inputs, wrt);
}

template <int dim, bool entvar>
double FlowResidual<dim, entvar>::calcEntropy_(const MachInputs &inputs)
{
   setInputs(ent, inputs);  // needed to set parameters in integrators
   return calcOutput(ent, inputs);
}

template <int dim, bool entvar>
double FlowResidual<dim, entvar>::calcEntropyChange_(const MachInputs &inputs)
{
   setInputs(res, inputs);  // needed to set parameters in integrators
   Vector x;
   setVectorFromInputs(inputs, "state", x, false, true);
   Vector dxdt;
   setVectorFromInputs(inputs, "state_dot", dxdt, false, true);
   double dt = NAN;
   double time = NAN;
   setValueFromInputs(inputs, "time", time, true);
   setValueFromInputs(inputs, "dt", dt, true);

   auto form_inputs = MachInputs({{"state", x}, {"time", time + dt}});
   return calcFormOutput(res, form_inputs);

   // ParGridFunction state(&fes), dstate(&fes);
   // state.SetFromTrueDofs(x);
   // dstate.SetFromTrueDofs(dxdt);
   // DenseMatrix u_elem, res_elem;
   // Vector u_j, res_j;
   // Vector w_j(dim+2);
   // double loc_change = 0.0;
   // for (int i = 0; i < fes.GetNE(); i++)
   // {
   //    const FiniteElement *fe = fes.GetFE(i);
   //    const IntegrationRule *ir = &(fe->GetNodes());
   //    ElementTransformation *trans = fes.GetElementTransformation(i);
   //    state.GetVectorValues(*trans, *ir, u_elem);
   //    dstate.GetVectorValues(*trans, *ir, res_elem);
   //    for (int j = 0; j < ir->GetNPoints(); j++)
   //    {
   //       const IntegrationPoint &ip = ir->IntPoint(j);
   //       trans->SetIntPoint(&ip);
   //       u_elem.GetColumnReference(j, u_j);
   //       res_elem.GetColumnReference(j, res_j);
   //       calcEntropyVars<double, dim, entvar>(u_j.GetData(),
   //                                            w_j.GetData());
   //       loc_change -= ip.weight * trans->Weight() * (w_j * res_j);
   //    }
   // }
   // double ent_change = NAN;
   // MPI_Allreduce(
   //     &loc_change, &ent_change, 1, MPI_DOUBLE, MPI_SUM, fes.GetComm());
   // return ent_change;

   // cout << "getSize_() = " << getSize_() << endl;
   // cout << "x.Size() = " << x.Size() << endl;
   // // cout << "num nodes  = " << getSize_()/(dim+2) << endl;
   // for (int i = 0; i < x.Size() / (dim + 2); ++i)
   // {
   //    auto ptr = (dim + 2) * i;
   //    calcEntropyVars<double, dim, entvar>(x.GetData() + ptr,
   //                                         work.GetData() + ptr);
   // }

   // // minus sign needed since dxdt = -res
   // double loc_change = -mass.InnerProduct(work, dxdt);
   // double ent_change = 0.0;
   // MPI_Allreduce(&loc_change,
   //               &ent_change,
   //               1,
   //               MPI_DOUBLE,
   //               MPI_SUM,
   //               fes.GetComm());
   // return ent_change;

   // minus sign needed since dxdt = -res
   // return -InnerProduct(fes.GetComm(), work, dxdt);

   // TODO: The following should be sufficient for the dot product, and it
   // avoids computing the form output, but there is an outstanding bug
   // const int num_state = dim + 2;
   // Array<int> vdofs(num_state);
   // Vector ui, resi;
   // Vector wi(num_state);
   // double loc_change = 0.0;
   // cout << "norm(x) = " << x.Norml2() << endl;
   // cout << "norm(res) = " << dxdt.Norml2() << endl;
   // cout << "fes.GetNE() = " << fes.GetNE() << endl;
   // for (int i = 0; i < fes.GetNE(); i++)
   // {
   //    const auto *fe = fes.GetFE(i);
   //    int num_nodes = fe->GetDof();
   //    for (int j = 0; j < num_nodes; j++)
   //    {
   //       int offset = i * num_nodes * num_state + j * num_state;
   //       for (int k = 0; k < num_state; k++)
   //       {
   //          vdofs[k] = offset + k;
   //       }
   //       x.GetSubVector(vdofs, ui);
   //       dxdt.GetSubVector(vdofs, resi);
   //       calcEntropyVars<double, dim, entvar>(ui.GetData(),
   //                                            wi.GetData());
   //       cout << "wi * resi = " << wi * resi << endl;
   //       loc_change -= wi * resi;
   //    }
   // }
   // double ent_change = 0.0;
   // MPI_Allreduce(&loc_change,
   //               &ent_change,
   //               1,
   //               MPI_DOUBLE,
   //               MPI_SUM,
   //               fes.GetComm());
   // return ent_change;
}

template <int dim, bool entvar>
Operator *FlowResidual<dim, entvar>::getMassMatrix_(
    const nlohmann::json &options)
{
   if (mass_mat)
   {
      return mass_mat.get();
   }
   else
   {
      mass.Assemble(0);  // May want to consider AssembleDiagonal(Vector &diag)
      mass.Finalize(0);
      mass_mat.reset(mass.ParallelAssemble());
      return mass_mat.get();
   }
}

template <int dim, bool entvar>
Solver *FlowResidual<dim, entvar>::getPreconditioner_()
{
   if (prec == nullptr)
   {
      prec = constructPreconditioner(fes, this->options["lin-prec"]);
   }
   return prec.get();
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
            dt_local = min(dt_local,
                           cfl * dx * dx /
                               calcSpectralRadius<double, dim, entvar>(
                                   dxij, ui));  // extra dx is to normalize dxij
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
      calcConservativeVars<double, dim, entvar>(discrete.GetData(),
                                                qdiscrete.GetData());
      calcConservativeVars<double, dim, entvar>(exact.GetData(),
                                                qexact.GetData());
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

template <int dim, bool entvar>
MachOutput FlowResidual<dim, entvar>::constructOutput(
    const std::string &fun,
    const nlohmann::json &options)
{
   if (fun == "drag")
   {
      // drag on the specified boundaries
      auto bdrs = options["boundaries"].get<vector<int>>();
      Vector drag_dir(dim);
      drag_dir = 0.0;
      if (dim == 1)
      {
         drag_dir(0) = 1.0;
      }
      else
      {
         drag_dir(iroll) = cos(aoa_fs);
         drag_dir(ipitch) = sin(aoa_fs);
      }
      drag_dir *= 1.0 / pow(mach_fs, 2.0);  // to get non-dimensional Cd
      FunctionalOutput fun_out(fes, fields);
      if (viscous)
      {
         Vector q_ref(dim + 2);
         getFreeStreamState(q_ref);
         fun_out.addOutputBdrFaceIntegrator(new SurfaceForce<dim>(stack,
                                                                  fes.FEColl(),
                                                                  dim + 2,
                                                                  re_fs,
                                                                  pr_fs,
                                                                  q_ref,
                                                                  drag_dir,
                                                                  mu),
                                            std::move(bdrs));
      }
      else
      {
         fun_out.addOutputBdrFaceIntegrator(
             new PressureForce<dim, entvar>(stack, fes.FEColl(), drag_dir),
             std::move(bdrs));
      }
      return fun_out;
   }
   else if (fun == "lift")
   {
      // lift on the specified boundaries
      auto bdrs = options["boundaries"].get<vector<int>>();
      Vector lift_dir(dim);
      lift_dir = 0.0;
      if (dim == 1)
      {
         lift_dir(0) = 0.0;
      }
      else
      {
         lift_dir(iroll) = -sin(aoa_fs);
         lift_dir(ipitch) = cos(aoa_fs);
      }
      lift_dir *= 1.0 / pow(mach_fs, 2.0);  // to get non-dimensional Cl
      FunctionalOutput fun_out(fes, fields);
      if (viscous)
      {
         Vector q_ref(dim + 2);
         getFreeStreamState(q_ref);
         fun_out.addOutputBdrFaceIntegrator(new SurfaceForce<dim>(stack,
                                                                  fes.FEColl(),
                                                                  dim + 2,
                                                                  re_fs,
                                                                  pr_fs,
                                                                  q_ref,
                                                                  lift_dir,
                                                                  mu),
                                            std::move(bdrs));
      }
      else
      {
         fun_out.addOutputBdrFaceIntegrator(
             new PressureForce<dim, entvar>(stack, fes.FEColl(), lift_dir),
             std::move(bdrs));
      }
      return fun_out;
   }
   else if (fun == "entropy")
   {
      // global entropy
      EntropyOutput<FlowResidual<dim, entvar>> fun_out(*this);
      return fun_out;
   }
   else if (fun == "boundary-entropy")
   {
      // weighted entropy over specified boundaries
      auto bdrs = options["boundaries"].get<vector<int>>();
      // define the scaling function for the entropy
      std::function<double(double, const Vector &, const Vector &)> scale =
          squaredExponential;
      // Should pass in xc and len...
      double len = 0.1;
      Vector xc({0.5, 0.0});
      FunctionalOutput fun_out(fes, fields);
      fun_out.addOutputBdrFaceIntegrator(
          new BoundaryEntropy<dim, entvar>(stack, fes.FEColl(), scale, xc, len),
          std::move(bdrs));
      return fun_out;
   }
   else
   {
      throw MachException("Output with name " + fun +
                          " not supported by "
                          "FlowResidual!\n");
   }
}

template <int dim, bool entvar>
void FlowResidual<dim, entvar>::getFreeStreamState(mfem::Vector &qfar)
{
   getFreeStreamQ<double, dim>(mach_fs, aoa_fs, iroll, ipitch, qfar.GetData());
}

// explicit instantiation
template class FlowResidual<1, true>;
template class FlowResidual<1, false>;
template class FlowResidual<2, true>;
template class FlowResidual<2, false>;
template class FlowResidual<3, true>;
template class FlowResidual<3, false>;

}  // namespace mach
