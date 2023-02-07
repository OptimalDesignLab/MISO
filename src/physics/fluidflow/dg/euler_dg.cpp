#include <cmath>
#include <memory>

#include "diag_mass_integ.hpp"
#include "euler_integ_dg.hpp"
#include "euler_integ.hpp"
#include "functional_output.hpp"
#include "sbp_fe.hpp"
#include "utils.hpp"

#include "euler_dg.hpp"

using namespace mfem;
using namespace std;

namespace mach
{
template <int dim, bool entvar>
EulerDGSolver<dim, entvar>::EulerDGSolver(const nlohmann::json &json_options,
                                          unique_ptr<mfem::Mesh> smesh,
                                          MPI_Comm comm)
 : AbstractSolver(json_options, move(smesh), comm)
{
   if (entvar)
   {
      *out << "The state variables are the entropy variables." << endl;
   }
   else
   {
      *out << "The state variables are the conservative variables." << endl;
   }
   // define free-stream parameters; may or may not be used, depending on case
   mach_fs = options["flow-param"]["mach"].template get<double>();
   aoa_fs = options["flow-param"]["aoa"].template get<double>() * M_PI / 180;
   iroll = options["flow-param"]["roll-axis"].template get<int>();
   ipitch = options["flow-param"]["pitch-axis"].template get<int>();
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
}

template <int dim, bool entvar>
void EulerDGSolver<dim, entvar>::constructForms()
{
   if (gd)
   {
      res.reset(new NonlinearFormType(fes_gd.get()));
      if ((entvar) && (!options["time-dis"]["steady"].template get<bool>()))
      {
         nonlinear_mass.reset(new NonlinearFormType(fes_gd.get()));
         mass.reset();
      }
      else
      {
         mass.reset(new BilinearFormType(fes_gd.get()));
         nonlinear_mass.reset();
      }
      ent.reset(new NonlinearFormType(fes_gd.get()));
   }
   else
   {
      res.reset(new NonlinearFormType(fes.get()));
      if ((entvar) && (!options["time-dis"]["steady"].template get<bool>()))
      {
         nonlinear_mass.reset(new NonlinearFormType(fes.get()));
         mass.reset();
      }
      else
      {
         mass.reset(new BilinearFormType(fes.get()));
         nonlinear_mass.reset();
      }
      ent.reset(new NonlinearFormType(fes.get()));
   }
}

template <int dim, bool entvar>
void EulerDGSolver<dim, entvar>::addMassIntegrators(double alpha)
{
   mass->AddDomainIntegrator(new DGMassIntegrator(num_state));
}

template <int dim, bool entvar>
void EulerDGSolver<dim, entvar>::addNonlinearMassIntegrators(double alpha)
{
   nonlinear_mass->AddDomainIntegrator(
       new MassIntegrator<dim, entvar>(diff_stack, alpha));
}

template <int dim, bool entvar>
void EulerDGSolver<dim, entvar>::addResVolumeIntegrators(double alpha, double &diff_coeff)
{
   // TODO: should decide between one-point and two-point fluxes using options
   // GridFunction x(fes.get());
   // ParCentGridFunction x(fes_gd.get());
   res->AddDomainIntegrator(new EulerDGIntegrator<dim>(diff_stack, alpha));
   res->AddDomainIntegrator(new EulerDiffusionIntegrator<dim>(diff_coeff, alpha));
   //double area;
   // area = res->GetEnergy(x);
   //  cout << "exact area: " << (5.45*5.45 - 0.25)*M_PI << endl;
   // cout << "calculated area: " << area << endl;
   // cout << "airfoil area " << endl;
   // cout << abs(M_PI * 900 - area) << endl;
   // add the LPS stabilization
   // auto lps_coeff = options["space-dis"]["lps-coeff"].template get<double>();
   // res->AddDomainIntegrator(
   //     new EntStableLPSIntegrator<dim, entvar>(diff_stack, alpha,
   //     lps_coeff));
   if (options["flow-param"]["inviscid-mms"].template get<bool>())
   {
      if (dim != 2)
      {
         throw MachException("Inviscid MMS problem only available for 2D!");
      }
      res->AddDomainIntegrator(
          new EulerMMSIntegrator<dim, entvar>(diff_stack, -alpha));
   }
   if (options["flow-param"]["potential-mms"].template get<bool>())
   {
      if (dim != 2)
      {
         throw MachException("Potential MMS problem only available for 2D!");
      }
      res->AddDomainIntegrator(
          new PotentialMMSIntegrator<dim, entvar>(diff_stack, -alpha));
   }
}

template <int dim, bool entvar>
void EulerDGSolver<dim, entvar>::addResBoundaryIntegrators(double alpha)
{
   auto &bcs = options["bcs"];
   int idx = 0;
   if (bcs.find("vortex") != bcs.end())
   {  // isentropic vortex BC
      if (dim != 2)
      {
         throw MachException(
             "EulerDGSolver::addBoundaryIntegrators(alpha)\n"
             "\tisentropic vortex BC must use 2D mesh!");
      }
      vector<int> tmp = bcs["vortex"].template get<vector<int>>();
      bndry_marker[idx].SetSize(tmp.size(), 0);
      bndry_marker[idx].Assign(tmp.data());
      res->AddBdrFaceIntegrator(
          new DGIsentropicVortexBC<dim, entvar>(diff_stack, fec.get(), alpha),
          bndry_marker[idx]);
      idx++;
   }

   if (bcs.find("slip-wall") != bcs.end())
   {  // slip-wall boundary condition
      vector<int> tmp = bcs["slip-wall"].template get<vector<int>>();
      bndry_marker[idx].SetSize(tmp.size(), 0);
      bndry_marker[idx].Assign(tmp.data());
      res->AddBdrFaceIntegrator(
          new DGSlipWallBC<dim, entvar>(diff_stack, fec.get(), alpha),
          bndry_marker[idx]);
      idx++;
      // GridFunction x(fes.get());
      // double peri_circle;
      // peri_circle = res->GetEnergy(x);
      // cout << "exact circle perimeter: " << M_PI << endl;
      // cout << "calculated circle perimeter: " << peri_circle << endl;
   }

   if (bcs.find("far-field") != bcs.end())
   {
      GridFunction x(fes.get());
      // far-field boundary conditions
      vector<int> tmp = bcs["far-field"].template get<vector<int>>();
      mfem::Vector qfar(dim + 2);
      getFreeStreamState(qfar);
      bndry_marker[idx].SetSize(tmp.size(), 0);
      bndry_marker[idx].Assign(tmp.data());
      res->AddBdrFaceIntegrator(
          new DGFarFieldBC<dim, entvar>(diff_stack, fec.get(), qfar, alpha),
          bndry_marker[idx]);
      idx++;
      // double peri_far;
      // peri_far = res->GetEnergy(x);
      // cout << "farfield perimeter: " << 2 * M_PI * 30 << endl;
      // cout << "calculated perimeter: " << peri_far << endl;
      // cout << "error: " << endl;
      // cout << abs(2 * M_PI * 30 - peri_far) << endl;
   }
   if (bcs.find("potential-flow") != bcs.end())
   {
      // far-field boundary conditions
      vector<int> tmp = bcs["potential-flow"].template get<vector<int>>();
      bndry_marker[idx].SetSize(tmp.size(), 0);
      bndry_marker[idx].Assign(tmp.data());
      res->AddBdrFaceIntegrator(
          new DGPotentialFlowBC<dim, entvar>(diff_stack, fec.get(), alpha),
          bndry_marker[idx]);
      idx++;
   }
   if (bcs.find("inviscid-mms") != bcs.end())
   {
      // viscous MMS boundary conditions
      vector<int> tmp = bcs["inviscid-mms"].template get<vector<int>>();
      bndry_marker[idx].SetSize(tmp.size(), 0);
      bndry_marker[idx].Assign(tmp.data());
      res->AddBdrFaceIntegrator(
          new InviscidDGExactBC<dim, entvar>(
              diff_stack, fec.get(), inviscidDGMMSExact, alpha),
          this->bndry_marker[idx]);
      idx++;
   }
}
template <int dim, bool entvar>
void EulerDGSolver<dim, entvar>::addResInterfaceIntegrators(double alpha)
{
   // add the integrators based on if discretization is continuous or discrete
   auto diss_coeff = options["space-dis"]["iface-coeff"].template get<double>();
   res->AddInteriorFaceIntegrator(new DGInterfaceIntegrator<dim, entvar>(
       diff_stack, diss_coeff, fec.get(), alpha));
}

template <int dim, bool entvar>
void EulerDGSolver<dim, entvar>::addEntVolumeIntegrators()
{
   ent->AddDomainIntegrator(new EntropyIntegrator<dim, entvar>(diff_stack));
}

template <int dim, bool entvar>
void EulerDGSolver<dim, entvar>::initialHook(const ParGridFunction &state)
{
   if (options["time-dis"]["steady"].template get<bool>())
   {
      // res_norm0 is used to compute the time step in PTC
      res_norm0 = calcResidualNorm(state);
   }
   // TODO: this should only be output if necessary
   // double entropy = ent->GetEnergy(state);
   // *out << "before time stepping, entropy is " << entropy << endl;
   // remove("entropylog.txt");
   // entropylog.open("entropylog.txt", fstream::app);
   // entropylog << setprecision(14);
}
template <int dim, bool entvar>
void EulerDGSolver<dim, entvar>::initialHook(const ParCentGridFunction &state)
{
   if (options["time-dis"]["steady"].template get<bool>())
   {
      // res_norm0 is used to compute the time step in PTC
      res_norm0 = calcResidualNorm(state);
   }
   // TODO: this should only be output if necessary
   GridFunType u_state(fes.get());
   fes_gd->GetProlongationMatrix()->Mult(state, u_state);
   // double entropy = ent->GetEnergy(u_state);
   // *out << "before time stepping, entropy is " << entropy << endl;
   // remove("entropylog.txt");
   // entropylog.open("entropylog.txt", fstream::app);
   // entropylog << setprecision(14);
}
template <int dim, bool entvar>
void EulerDGSolver<dim, entvar>::iterationHook(int iter,
                                               double t,
                                               double dt,
                                               const ParGridFunction &state)
{
   double entropy = ent->GetEnergy(state);
   entropylog << t << ' ' << entropy << endl;
}

template <int dim, bool entvar>
bool EulerDGSolver<dim, entvar>::iterationExit(
    int iter,
    double t,
    double t_final,
    double dt,
    const ParGridFunction &state) const
{
   if (options["time-dis"]["steady"].template get<bool>())
   {
      // use tolerance options for Newton's method
      double norm = calcResidualNorm(state);
      if (norm <= options["time-dis"]["steady-abstol"].template get<double>())
      {
         return true;
      }
      if (norm <=
          res_norm0 *
              options["time-dis"]["steady-reltol"].template get<double>())
      {
         return true;
      }
      return false;
   }
   else
   {
      return AbstractSolver::iterationExit(iter, t, t_final, dt, state);
   }
}

template <int dim, bool entvar>
bool EulerDGSolver<dim, entvar>::iterationExit(
    int iter,
    double t,
    double t_final,
    double dt,
    const ParCentGridFunction &state) const
{
   if (options["time-dis"]["steady"].template get<bool>())
   {
      // use tolerance options for Newton's method
      double norm = calcResidualNorm(state);
      if (norm <= options["time-dis"]["steady-abstol"].template get<double>())
      {
         return true;
      }
      if (norm <=
          res_norm0 *
              options["time-dis"]["steady-reltol"].template get<double>())
      {
         return true;
      }
      return false;
   }
   else
   {
      return AbstractSolver::iterationExit(iter, t, t_final, dt, state);
   }
}

template <int dim, bool entvar>
void EulerDGSolver<dim, entvar>::terminalHook(int iter,
                                              double t_final,
                                              const ParGridFunction &state)
{
   double entropy = ent->GetEnergy(state);
   entropylog << t_final << ' ' << entropy << endl;
   entropylog.close();
}

template <int dim, bool entvar>
void EulerDGSolver<dim, entvar>::addOutput(const std::string &fun,
                                           const nlohmann::json &options)
{
   if (fun == "drag")
   {
      // drag on the specified boundaries
      auto bdrs = options["boundaries"].template get<vector<int>>();

      mfem::Vector drag_dir(dim);
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

      FunctionalOutput out(*fes, res_fields);
      out.addOutputBdrFaceIntegrator(
          new DGPressureForce<dim, entvar>(diff_stack, fec.get(), drag_dir),
          std::move(bdrs));
      outputs.emplace(fun, std::move(out));
   }
   else if (fun == "lift")
   {
      // lift on the specified boundaries
      auto bdrs = options["boundaries"].template get<vector<int>>();

      mfem::Vector lift_dir(dim);
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

      FunctionalOutput out(*fes, res_fields);
      out.addOutputBdrFaceIntegrator(
          new DGPressureForce<dim, entvar>(diff_stack, fec.get(), lift_dir),
          std::move(bdrs));
      outputs.emplace(fun, std::move(out));
   }
   else if (fun == "entropy")
   {
      // integral of entropy over the entire volume domain
      // FunctionalOutput out(*fes, res_fields);
      // out.addOutputDomainIntegrator(
      //     new EntropyIntegrator<dim, entvar>(diff_stack));
      // outputs.emplace(fun, std::move(out));
   }
   else
   {
      throw MachException("Output with name " + fun +
                          " not supported by "
                          "EulerDGSolver!\n");
   }
}

template <int dim, bool entvar>
double EulerDGSolver<dim, entvar>::calcStepSize(
    int iter,
    double t,
    double t_final,
    double dt_old,
    const ParGridFunction &state) const
{
   if (options["time-dis"]["steady"].template get<bool>())
   {
      // ramp up time step for pseudo-transient continuation
      // TODO: the l2 norm of the weak residual is probably not ideal here
      // A better choice might be the l1 norm
      double res_norm = calcResidualNorm(state);
      double exponent = options["time-dis"]["res-exp"];
      double dt = options["time-dis"]["dt"].template get<double>() *
                  pow(res_norm0 / res_norm, exponent);
      return max(dt, dt_old);
   }
   if (!options["time-dis"]["const-cfl"].template get<bool>())
   {
      return options["time-dis"]["dt"].template get<double>();
   }
   // Otherwise, use a constant CFL condition
   auto cfl = options["time-dis"]["cfl"].template get<double>();
   Vector q(dim + 2);
   auto calcSpect = [&q](const double *dir, const double *u)
   {
      if (entvar)
      {
         calcConservativeVars<double, dim>(u, q);
         return calcSpectralRadius<double, dim>(dir, q);
      }
      else
      {
         return calcSpectralRadius<double, dim>(dir, u);
      }
   };
   double dt_local = 1e100;
   Vector xi(dim);
   Vector dxij(dim);
   Vector ui;
   Vector dxidx;
   DenseMatrix uk;
   DenseMatrix adjJt(dim);
   for (int k = 0; k < fes->GetNE(); k++)
   {
      // get the element, its transformation, and the state values on element
      const FiniteElement *fe = fes->GetFE(k);
      const IntegrationRule *ir = &(fe->GetNodes());
      ElementTransformation *trans = fes->GetElementTransformation(k);
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
                        calcSpect(dxij, ui));  // extra dx is to normalize dxij
         }
      }
   }
   double dt_min = NAN;
   MPI_Allreduce(&dt_local, &dt_min, 1, MPI_DOUBLE, MPI_MIN, comm);
   return dt_min;
}

template <int dim, bool entvar>
double EulerDGSolver<dim, entvar>::calcStepSize(
    int iter,
    double t,
    double t_final,
    double dt_old,
    const ParCentGridFunction &state) const
{
   if (options["time-dis"]["steady"].template get<bool>())
   {
      // ramp up time step for pseudo-transient continuation
      // TODO: the l2 norm of the weak residual is probably not ideal here
      // A better choice might be the l1 norm
      double res_norm = calcResidualNorm(state);
      double exponent = options["time-dis"]["res-exp"];
      double dt = options["time-dis"]["dt"].template get<double>() *
                  pow(res_norm0 / res_norm, exponent);
      return max(dt, dt_old);
   }
   if (!options["time-dis"]["const-cfl"].template get<bool>())
   {
      return options["time-dis"]["dt"].template get<double>();
   }
   // Otherwise, use a constant CFL condition
   auto cfl = options["time-dis"]["cfl"].template get<double>();
   Vector q(dim + 2);
   auto calcSpect = [&q](const double *dir, const double *u)
   {
      if (entvar)
      {
         calcConservativeVars<double, dim>(u, q);
         return calcSpectralRadius<double, dim>(dir, q);
      }
      else
      {
         return calcSpectralRadius<double, dim>(dir, u);
      }
   };
   double dt_local = 1e100;
   Vector xi(dim);
   Vector dxij(dim);
   Vector ui;
   Vector dxidx;
   DenseMatrix uk;
   DenseMatrix adjJt(dim);
   for (int k = 0; k < fes->GetNE(); k++)
   {
      // get the element, its transformation, and the state values on element
      const FiniteElement *fe = fes->GetFE(k);
      const IntegrationRule *ir = &(fe->GetNodes());
      ElementTransformation *trans = fes->GetElementTransformation(k);
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
                        calcSpect(dxij, ui));  // extra dx is to normalize dxij
         }
      }
   }
   double dt_min = NAN;
   MPI_Allreduce(&dt_local, &dt_min, 1, MPI_DOUBLE, MPI_MIN, comm);
   return dt_min;
}

template <int dim, bool entvar>
void EulerDGSolver<dim, entvar>::getFreeStreamState(mfem::Vector &q_ref)
{
   q_ref = 0.0;
   q_ref(0) = 1.0;
   if (dim == 1)
   {
      q_ref(1) = q_ref(0) * mach_fs;  // ignore angle of attack
   }
   else
   {
      q_ref(iroll + 1) = q_ref(0) * mach_fs * cos(aoa_fs);
      q_ref(ipitch + 1) = q_ref(0) * mach_fs * sin(aoa_fs);
   }
   q_ref(dim + 1) =
       1.0 / (euler::gamma * euler::gami) + 0.5 * mach_fs * mach_fs;
}

template <int dim, bool entvar>
double EulerDGSolver<dim, entvar>::calcConservativeVarsL2Error(
    void (*u_exact)(const mfem::Vector &, mfem::Vector &),
    int entry)
{
   // This lambda function computes the error at a node
   // Beware: this is not particularly efficient, given the conditionals
   // Also **NOT thread safe!**
   Vector qdiscrete(dim + 2);
   Vector qexact(dim + 2);  // define here to avoid reallocation
   auto node_error = [&](const Vector &discrete, const Vector &exact) -> double
   {
      if (entvar)
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

   VectorFunctionCoefficient exsol(num_state, u_exact);
   DenseMatrix vals;
   DenseMatrix exact_vals;
   Vector u_j;
   Vector exsol_j;
   double loc_norm = 0.0;
   for (int i = 0; i < fes->GetNE(); i++)
   {
      const FiniteElement *fe = fes->GetFE(i);
      // const IntegrationRule *ir = &(fe->GetNodes());
      const IntegrationRule *ir;
      int intorder = fe->GetOrder();
      ir = &(IntRules.Get(fe->GetGeomType(), intorder));
      ElementTransformation *T = fes->GetElementTransformation(i);
      u->GetVectorValues(*T, *ir, vals);
      exsol.Eval(exact_vals, *T, *ir);
      for (int j = 0; j < ir->GetNPoints(); j++)
      {
         const IntegrationPoint &ip = ir->IntPoint(j);
         T->SetIntPoint(&ip);
         vals.GetColumnReference(j, u_j);
         exact_vals.GetColumnReference(j, exsol_j);
         loc_norm += ip.weight * T->Weight() * node_error(u_j, exsol_j);
         // std::cout << "ip.w " << ip.weight << std::endl;
      }

      // std::cout << "loc_norm " << loc_norm << std::endl;
   }
   double norm = NAN;
   MPI_Allreduce(&loc_norm, &norm, 1, MPI_DOUBLE, MPI_SUM, comm);
   if (norm < 0.0)  // This was copied from mfem...should not happen for us
   {
      return -sqrt(-norm);
   }
   return sqrt(norm);
}

template <int dim, bool entvar>
void EulerDGSolver<dim, entvar>::convertToEntvar(mfem::Vector &state)
{
   if (entvar)
   {
      return;
   }
   else
   {
      mfem::Array<int> vdofs(num_state);
      Vector el_con;
      Vector el_ent;
      for (int i = 0; i < fes->GetNE(); i++)
      {
         const auto *fe = fes->GetFE(i);
         int num_nodes = fe->GetDof();
         for (int j = 0; j < num_nodes; j++)
         {
            int offset = i * num_nodes * num_state + j * num_state;
            for (int k = 0; k < num_state; k++)
            {
               vdofs[k] = offset + k;
            }
            u->GetSubVector(vdofs, el_con);
            calcEntropyVars<double, dim>(el_con.GetData(), el_ent.GetData());
            state.SetSubVector(vdofs, el_ent);
         }
      }
   }
}

template <int dim, bool entvar>
void EulerDGSolver<dim, entvar>::setSolutionError(
    void (*u_exact)(const mfem::Vector &, mfem::Vector &))
{
   VectorFunctionCoefficient exsol(num_state, u_exact);
   GridFunType ue(fes.get());
   ue.ProjectCoefficient(exsol);
   // TODO: are true DOFs necessary here?
   HypreParVector *u_true = u->GetTrueDofs();
   HypreParVector *ue_true = ue.GetTrueDofs();
   *u_true -= *ue_true;
   u->SetFromTrueDofs(*u_true);
}

/// MMS for checking euler solver
void inviscidDGMMSExact(const mfem::Vector &x, mfem::Vector &q)
{
   q.SetSize(4);
   Vector u(4);
   const double rho0 = 1.0;
   const double rhop = 0.05;
   const double u0 = 0.5;
   const double up = 0.05;
   const double T0 = 1.0;
   const double Tp = 0.05;
   const double scale = 1.0;
   const double trans = 0.0;
   /// define the exact solution
   double rho = rho0 + rhop * pow(sin(M_PI * (x(0) + trans) / scale), 2) *
                           sin(M_PI * (x(1) + trans) / scale);
   double ux =
       4.0 * u0 * ((x(1) + trans) / scale) * (1.0 - (x(1) + trans) / scale) +
       (up * sin(2.0 * M_PI * (x(1) + trans) / scale) *
        pow(sin(M_PI * (x(0) + trans) / scale), 2));
   double uy = -up * pow(sin(2.0 * M_PI * (x(0) + trans) / scale), 2) *
               sin(M_PI * (x(1) + trans) / scale);
   double T = T0 + Tp * (pow((x(0) + trans) / scale, 4) -
                         (2.0 * pow((x(0) + trans) / scale, 3)) +
                         pow((x(0) + trans) / scale, 2) +
                         pow((x(1) + trans) / scale, 4) -
                         (2.0 * pow((x(1) + trans) / scale, 3)) +
                         pow((x(1) + trans) / scale, 2));
   double p = rho * T;
   double e = (p / (euler::gamma - 1)) + 0.5 * rho * (ux * ux + uy * uy);
   u(0) = rho;
   u(1) = rho * ux;  // multiply by rho ?
   u(2) = rho * uy;
   u(3) = e;
   q = u;
}
// explicit instantiation
template class EulerDGSolver<1, true>;
template class EulerDGSolver<1, false>;
template class EulerDGSolver<2, true>;
template class EulerDGSolver<2, false>;
template class EulerDGSolver<3, true>;
template class EulerDGSolver<3, false>;

}  // namespace mach
