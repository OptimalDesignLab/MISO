#include "mfem.hpp"

#include "diag_mass_integ.hpp"
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
                                        adept::Stack &diff_stack,
                                        ostream &outstream)
 : out(outstream),
   fes(fespace),
   stack(diff_stack),
   fields(std::make_unique<
          std::unordered_map<std::string, mfem::ParGridFunction>>()),
   res(fes, *fields),
   mass(&fespace),
   ent(fes, *fields),
   work(getSize(res))
{
   setOptions_(options);

   if (!options.contains("flow-param") || !options.contains("space-dis"))
   {
      throw MachException(
          "FlowResidual::addFlowIntegrators: options must"
          "contain flow-param and space-dis!\n");
   }
   nlohmann::json flow = options["flow-param"];
   nlohmann::json space_dis = options["space-dis"];
   addFlowDomainIntegrators(flow, space_dis);
   addFlowInterfaceIntegrators(flow, space_dis);
   if (options.contains("bcs"))
   {
      nlohmann::json bcs = options["bcs"];
      addFlowBoundaryIntegrators(flow, space_dis, bcs);
   }

   // set up the mass bilinear form, but do not construct the matrix unless
   // necessary (see getMassMatrix_)
   const char *name = fes.FEColl()->Name();
   if ((strncmp(name, "SBP", 3) == 0) || (strncmp(name, "DSBP", 4) == 0))
   {
      mass.AddDomainIntegrator(new DiagMassIntegrator(fes.GetVDim()));
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
      getFreeStreamQ<double, dim, entvar>(
          mach_fs, aoa_fs, iroll, ipitch, qfar.GetData());
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
      auto pump = [](double t, const mfem::Vector &x, mfem::Vector &q)
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
   auto &y = work;
   add(x, dt, dxdt, y);
   auto form_inputs = MachInputs({{"state", y}, {"time", time + dt}});
   return calcFormOutput(res, form_inputs);
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
Solver *FlowResidual<dim, entvar>::getPreconditioner_(
    const nlohmann::json &prec_options)
{
   std::string prec_type = prec_options["type"].get<std::string>();
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
      HYPRE_ILUSetType(*ilu, prec_options["ilu-type"].get<int>());
      HYPRE_ILUSetLevelOfFill(*ilu, prec_options["lev-fill"].get<int>());
      HYPRE_ILUSetLocalReordering(*ilu, prec_options["ilu-reorder"].get<int>());
      HYPRE_ILUSetPrintLevel(*ilu, prec_options["printlevel"].get<int>());
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
      ams->SetPrintLevel(prec_options["printlevel"].get<int>());
      ams->SetSingularProblem();
   }
   else if (prec_type == "hypreboomeramg")
   {
      prec = std::make_unique<HypreBoomerAMG>();
      auto *amg = dynamic_cast<HypreBoomerAMG *>(prec.get());
      amg->SetPrintLevel(prec_options["printlevel"].get<int>());
   }
   else if (prec_type == "blockilu")
   {
      prec = std::make_unique<BlockILU>(fes.GetVDim());
   }
   else
   {
      throw MachException(
          "Unsupported preconditioner type!\n"
          "\tavilable options are: HypreEuclid, HypreILU, HypreAMS,"
          " HypreBoomerAMG.\n");
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
      FunctionalOutput fun_out(fes, *fields);
      fun_out.addOutputBdrFaceIntegrator(
          new PressureForce<dim, entvar>(stack, fes.FEColl(), drag_dir),
          std::move(bdrs));
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
      FunctionalOutput fun_out(fes, *fields);
      fun_out.addOutputBdrFaceIntegrator(
          new PressureForce<dim, entvar>(stack, fes.FEColl(), lift_dir),
          std::move(bdrs));
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
      FunctionalOutput fun_out(fes, *fields);
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

// explicit instantiation
template class FlowResidual<1, true>;
template class FlowResidual<1, false>;
template class FlowResidual<2, true>;
template class FlowResidual<2, false>;
template class FlowResidual<3, true>;
template class FlowResidual<3, false>;

}  // namespace mach
