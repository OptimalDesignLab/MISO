#include <memory>

#include "navier_stokes.hpp"
#include "navier_stokes_integ.hpp"

using namespace mfem;
using namespace std;

namespace mach
{

template <int dim, bool entvar>
NavierStokesSolver<dim, entvar>::NavierStokesSolver(
   const nlohmann::json &json_options, unique_ptr<mfem::Mesh> smesh)
    : EulerSolver<dim, entvar>(json_options, move(smesh))
{
   // define NS-related parameters; may or may not be used, depending on case
   re_fs = this->options["flow-param"]["Re"].template get<double>();
   pr_fs = this->options["flow-param"]["Pr"].template get<double>();

   // Note: the viscous terms are added to the semi-linear form `res` via
   // virtual function calls to addVolumeIntegrators(), addBoundaryIntegrators,
   // and addInterfaceIntegrators(). 

   // add the output functional QoIs 
   //addOutputs(dim);
}

template <int dim, bool entvar>
void NavierStokesSolver<dim, entvar>::addResVolumeIntegrators(double alpha)
{
   // add base class integrators
   EulerSolver<dim, entvar>::addResVolumeIntegrators(alpha);
   cout << "Inside NS add volume integrators" << endl;
   // now add NS integrators
   double mu = this->options["flow-param"]["mu"].template get<double>();
   this->res->AddDomainIntegrator(new ESViscousIntegrator<dim>(
       this->diff_stack, re_fs, pr_fs, mu, alpha));

   if (this->options["flow-param"]["viscous-mms"].template get<bool>())
   {
      if (dim != 2)
      {
         throw MachException("Viscous MMS problem only available for 2D!");
      }
      this->res->AddDomainIntegrator(
          new NavierStokesMMSIntegrator(re_fs, pr_fs));
   }
}

template <int dim, bool entvar>
void NavierStokesSolver<dim, entvar>::addResBoundaryIntegrators(double alpha)
{
   cout << "Inside NS add boundary integrators" << endl;
   auto &bcs = this->options["bcs"];
   double mu = this->options["flow-param"]["mu"].template get<double>();
   int idx = 0;
   if (bcs.find("slip-wall") != bcs.end())
   { // slip-wall boundary condition
      vector<int> tmp = bcs["slip-wall"].template get<vector<int>>();
      this->bndry_marker[idx].SetSize(tmp.size(), 0);
      this->bndry_marker[idx].Assign(tmp.data());
      this->res->AddBdrFaceIntegrator(
          new ViscousSlipWallBC<dim>(this->diff_stack, this->fec.get(),
                                     re_fs, pr_fs, mu, alpha),
          this->bndry_marker[idx]);
      idx++;
   }
   if (bcs.find("no-slip-adiabatic") != bcs.end())
   {
      vector<int> tmp = bcs["no-slip-adiabatic"].template get<vector<int>>();
      this->bndry_marker[idx].SetSize(tmp.size(), 0);
      this->bndry_marker[idx].Assign(tmp.data());
      // reference state needed by penalty flux
      Vector q_ref(dim+2);
      this->getFreeStreamState(q_ref);
      // Add the adiabatic flux BC
      this->res->AddBdrFaceIntegrator(
          new NoSlipAdiabaticWallBC<dim>(
              this->diff_stack, this->fec.get(), re_fs, pr_fs, q_ref, mu, alpha),
          this->bndry_marker[idx]);
      idx++;
   }
   if (bcs.find("viscous-inflow") != bcs.end())
   {
      vector<int> tmp = bcs["viscous-inflow"].template get<vector<int>>();
      this->bndry_marker[idx].SetSize(tmp.size(), 0);
      this->bndry_marker[idx].Assign(tmp.data());
      // get the in-flow state needed by the integrator
      Vector q_in(dim+2);
      getViscousInflowState(q_in);
      // Add the viscous-inflow integrator
      this->res->AddBdrFaceIntegrator(
          new ViscousInflowBC<dim>(this->diff_stack, this->fec.get(),
                                   re_fs, pr_fs, q_in, mu, alpha),
          this->bndry_marker[idx]);
      idx++;
   }
   if (bcs.find("viscous-outflow") != bcs.end())
   {
      vector<int> tmp = bcs["viscous-outflow"].template get<vector<int>>();
      this->bndry_marker[idx].SetSize(tmp.size(), 0);
      this->bndry_marker[idx].Assign(tmp.data());
      // get the out-flow state needed by the integrator
      Vector q_out(dim+2);
      getViscousOutflowState(q_out);
      // Add the viscous-inflow integrator
      this->res->AddBdrFaceIntegrator(
          new ViscousOutflowBC<dim>(this->diff_stack, this->fec.get(),
                                   re_fs, pr_fs, q_out, mu, alpha),
          this->bndry_marker[idx]);
      idx++;
   }
   if (bcs.find("far-field") != bcs.end())
   { 
      // far-field boundary conditions
      vector<int> tmp = bcs["far-field"].template get<vector<int>>();
      mfem::Vector qfar(dim+2);
      this->getFreeStreamState(qfar);
      this->bndry_marker[idx].SetSize(tmp.size(), 0);
      this->bndry_marker[idx].Assign(tmp.data());
      this->res->AddBdrFaceIntegrator(
          new FarFieldBC<dim, entvar>(this->diff_stack, this->fec.get(), qfar,
                              alpha),
          this->bndry_marker[idx]);
      idx++;
   }
   if (bcs.find("viscous-shock") != bcs.end())
   { 
      // viscous shock boundary conditions
      vector<int> tmp = bcs["viscous-shock"].template get<vector<int>>();
      this->bndry_marker[idx].SetSize(tmp.size(), 0);
      this->bndry_marker[idx].Assign(tmp.data());
      this->res->AddBdrFaceIntegrator(
          new ViscousExactBC<dim>(this->diff_stack, this->fec.get(),
                                  re_fs, pr_fs, shockExact, mu, alpha),
          this->bndry_marker[idx]);
      idx++;
   }
   if (bcs.find("viscous-mms") != bcs.end())
   { 
      // viscous MMS boundary conditions
      vector<int> tmp = bcs["viscous-mms"].template get<vector<int>>();
      this->bndry_marker[idx].SetSize(tmp.size(), 0);
      this->bndry_marker[idx].Assign(tmp.data());
      this->res->AddBdrFaceIntegrator(
          new ViscousExactBC<dim>(this->diff_stack, this->fec.get(),
                                  re_fs, pr_fs, viscousMMSExact, mu, alpha),
          this->bndry_marker[idx]);
      idx++;
   }
}

template <int dim, bool entvar>
void NavierStokesSolver<dim, entvar>::addResInterfaceIntegrators(double alpha)
{
   // add base class integrators
   EulerSolver<dim, entvar>::addResInterfaceIntegrators(alpha);
}

template <int dim, bool entvar>
void NavierStokesSolver<dim, entvar>::addOutputs()
{
   cout << "Inside NS add Outputs" << endl;
   auto &fun = this->options["outputs"];
   auto &bcs = this->options["bcs"];
   double mu = this->options["flow-param"]["mu"].template get<double>();
   Vector q_ref(dim+2);
   this->getFreeStreamState(q_ref);
   int idx = 0;
   if (fun.find("drag") != fun.end())
   { 
      // drag on the specified boundaries
      vector<int> tmp = fun["drag"].template get<vector<int>>();
      this->output_bndry_marker[idx].SetSize(tmp.size(), 0);
      this->output_bndry_marker[idx].Assign(tmp.data());
      this->output.emplace("drag", this->fes.get());
      mfem::Vector drag_dir(dim);
      drag_dir = 0.0;
      if (dim == 1)
      {
         drag_dir(0) = 1.0;
      }
      else 
      {
         drag_dir(this->iroll) = cos(this->aoa_fs);
         drag_dir(this->ipitch) = sin(this->aoa_fs);
      }
      drag_dir *= 1.0 / pow(this->mach_fs, 2.0); // to get non-dimensional Cd
      this->output.at("drag").AddBdrFaceIntegrator(
          new SurfaceForce<dim>(this->diff_stack, this->fec.get(), dim + 2,
                                re_fs, pr_fs, q_ref, drag_dir, mu),
          this->output_bndry_marker[idx]);
      idx++;
   }
   if (fun.find("lift") != fun.end())
   { 
      // lift on the specified boundaries
      vector<int> tmp = fun["lift"].template get<vector<int>>();
      this->output_bndry_marker[idx].SetSize(tmp.size(), 0);
      this->output_bndry_marker[idx].Assign(tmp.data());
      this->output.emplace("lift", this->fes.get());
      mfem::Vector lift_dir(dim);
      lift_dir = 0.0;
      if (dim == 1)
      {
         lift_dir(0) = 0.0;
      }
      else
      {
         lift_dir(this->iroll) = -sin(this->aoa_fs);
         lift_dir(this->ipitch) = cos(this->aoa_fs);
      }
      lift_dir *= 1.0 / pow(this->mach_fs, 2.0); // to get non-dimensional Cl
      this->output.at("lift").AddBdrFaceIntegrator(
          new SurfaceForce<dim>(this->diff_stack, this->fec.get(), dim + 2,
                                re_fs, pr_fs, q_ref, lift_dir, mu),
          this->output_bndry_marker[idx]);
      idx++;
   }
   if (fun.find("entropy") != fun.end())
   {
      // integral of entropy over the entire volume domain
      this->output.emplace("entropy", this->fes.get());
      this->output.at("entropy").AddDomainIntegrator(
         new EntropyIntegrator<dim, entvar>(this->diff_stack));
   }
}

template <int dim, bool entvar>
void NavierStokesSolver<dim, entvar>::getViscousInflowState(Vector &q_in)
{
   vector<double> tmp = this->options["flow-param"]
                                     ["inflow-state"]
                                         .template get<vector<double>>();
   if (tmp.size() != dim+2)
   {
      throw MachException("inflow-state option has wrong number of entries"
                          " for problem dimension!");
   }
   for (int i = 0; i < dim+2; ++i)
   {
      q_in(i) = tmp[i];
   }
}

template <int dim, bool entvar>
void NavierStokesSolver<dim, entvar>::getViscousOutflowState(Vector &q_out)
{
   vector<double> tmp = this->options["flow-param"]
                                     ["outflow-state"]
                                         .template get<vector<double>>();
   if (tmp.size() != dim+2)
   {
      throw MachException("outflow-state option has wrong number of entries"
                          " for problem dimension!");
   }
   for (int i = 0; i < dim+2; ++i)
   {
      q_out(i) = tmp[i];
   }
}

template <int dim, bool entvar>
void NavierStokesSolver<dim, entvar>::solveEntBalanceAdjoint()
{
   double time_beg, time_end;
   time_beg = MPI_Wtime();

   // Use res_fun for the dJdu calculations; this way, we can verify that 
   // the entropy variables are not adjoints for non-ent stable codes
   NonlinearFormType res_fun(this->fes.get());
   res_fun.AddDomainIntegrator(
       new IsmailRoeIntegrator<dim, entvar>(this->diff_stack, 1.0));
   double lps_coeff = this->options["space-dis"]["lps-coeff"].template get<double>();
   res_fun.AddDomainIntegrator(
       new EntStableLPSIntegrator<dim, entvar>(this->diff_stack, 1.0, lps_coeff));
   double mu = this->options["flow-param"]["mu"].template get<double>();
   res_fun.AddDomainIntegrator(new ESViscousIntegrator<dim>(
       this->diff_stack, re_fs, pr_fs, mu, 1.0));

   // WARNING: this is not general!  It assumes only no-slip and far-field BCs,
   // and that they are added in a particular order
   Vector qfar(dim + 2);
   this->getFreeStreamState(qfar);
   res_fun.AddBdrFaceIntegrator(
       new NoSlipAdiabaticWallBC<dim>(
           this->diff_stack, this->fec.get(), re_fs, pr_fs, qfar, mu, 1.0),
       this->bndry_marker[0]);
   res_fun.AddBdrFaceIntegrator(
       new FarFieldBC<dim, entvar>(this->diff_stack, this->fec.get(), qfar,
                                   1.0),
       this->bndry_marker[1]);

   // Step 0: allocate the adjoint variable
   cout << "Step 0" << endl;
   this->adj.reset(new GridFunType(this->fes.get()));
   HypreParVector *adj_true = this->adj->GetTrueDofs();

   // Step 1: get the entropy variables
   cout << "Step 1" << endl;
   GridFunType w(this->fes.get());
   this->convertToEntvar(w);

   // Step 2: get the Jacobian and transpose it
   cout << "Step 2" << endl;
   HypreParVector *u_true = this->u->GetTrueDofs();
   Operator *jac = &(this->res->GetGradient(*u_true));
   Operator *jac_trans =
       dynamic_cast<const HypreParMatrix *>(jac)->Transpose();
   MFEM_VERIFY(jac_trans, "Jacobian must be a HypreParMatrix!");

   // Step 3: get the right-hand side vector, dJdu, and make an appropriate
   // alias to it, the state, and the adjoint
   // For the trivial functional, dJ/du = (dR/du)^*w, where R uses res_fun
   Operator *jac_entbal = &res_fun.GetGradient(*u_true);
   Operator *jac_entbal_trans =
       dynamic_cast<const HypreParMatrix *>(jac_entbal)->Transpose();
   MFEM_VERIFY(jac_entbal_trans, "Jacobian must be a HypreParMatrix!");
   std::unique_ptr<GridFunType> dJdu(new GridFunType(this->fes.get()));
   HypreParVector *dJdu_true = dJdu->GetTrueDofs();
   HypreParVector *w_true = w.GetTrueDofs();
   dynamic_cast<HypreParMatrix*>(jac_entbal_trans)->Mult(*w_true, *dJdu_true);

   // Step 4: Solve the adjoint problem
   cout << "Step 4" << endl;
   cout << "Solving adjoint problem:\n";
   unique_ptr<Solver> adj_prec = this->constructPreconditioner(this->options["adj-prec"]);
   cout << "Step 4a" << endl;
   unique_ptr<Solver> adj_solver = this->constructLinearSolver(this->options["adj-solver"],
                                                         *adj_prec);
   cout << "Step 4b" << endl;
   adj_solver->SetOperator(*jac_trans);
   cout << "Step 4c" << endl;
   adj_solver->Mult(*dJdu_true, *adj_true);

   // check that adjoint residual is small
   cout << "Step 5" << endl;
   std::unique_ptr<GridFunType> adj_res(new GridFunType(this->fes.get()));
   double res_norm = 0;
   HypreParVector *adj_res_true = adj_res->GetTrueDofs();
   jac_trans->Mult(*adj_true, *adj_res_true);
   *adj_res_true -= *dJdu_true;
   double loc_norm = (*adj_res_true)*(*adj_res_true);
   MPI_Allreduce(&loc_norm, &res_norm, 1, MPI_DOUBLE, MPI_SUM, this->comm);
   res_norm = sqrt(res_norm);
   cout << "Adjoint residual norm = " << res_norm << endl;
   this->adj->SetFromTrueDofs(*adj_true);
 
   time_end = MPI_Wtime();
   cout << "Time for solving adjoint is " << (time_end - time_beg) << endl;

   res_fun.Mult(*u_true, *dJdu_true);
   dJdu->SetFromTrueDofs(*dJdu_true);
   cout.precision(14);
   cout << "Entropy balance functional value = " << (*dJdu)*w << endl;
   w.Add(-1.0, *(this->adj));
   double error = sqrt(this->calcInnerProduct(w, w));

   cout << "L2 error in the entropy adjoint = " << error << endl;
}

// explicit instantiation
template class NavierStokesSolver<1>;
template class NavierStokesSolver<2>;
template class NavierStokesSolver<3>;

double shockEquation(double Re, double Ma, double v)
{
   double vf = (2.0 + euler::gami * Ma * Ma) / ((euler::gamma + 1) * Ma * Ma);
   double alpha = (8*euler::gamma)/(3*(euler::gamma+1)*Re*Ma);
   double r = (1 + vf) / (1 - vf);
   double a = abs((v - 1) * (v - vf));
   double b = (1 + vf) / (1 - vf);
   double c = abs((v - 1) / (v - vf));
   return 0.5 * alpha * (log(a) + b * log(c));
}

// Exact solution
void shockExact(const mfem::Vector &x, mfem::Vector& u)
{
   double Re = 10.0; // !!!!! Values from options file are ignored
   double Ma = 2.5;
   double vf = (2.0 + euler::gami*Ma*Ma)/((euler::gamma + 1)*Ma*Ma);
   double v;
   double ftol = 1e-10;
   double xtol = 1e-10;
   int maxiter = 50;
   if (x(0) < -1.25) 
   {
      v = 1.0;
   }
   else if (x(0) > 0.4) 
   {
      v = vf;
   }
   else
   {
      // define a lambda function for equation (7.5)
      auto func = [&](double vroot)
      {
         return x(0) - shockEquation(Re, Ma, vroot);
      };
      v = bisection(func, 1.0000001*vf, 0.9999999, ftol, xtol, maxiter);
   }
   double vel = v*Ma;
   u(0) = Ma/vel;  // rho*u = M_L
   u(1) = Ma;
   u(2) = 0.0;
   u(3) = 0.5 * (euler::gamma + 1) * vf * u(0) * Ma * Ma /
              (euler::gamma * euler::gami) +
          euler::gami * Ma * vel / (2 * euler::gamma);
}

// MMS Exact solution
void viscousMMSExact(const mfem::Vector &x, mfem::Vector& u)
{
   const double rho0 = 1.0;
   const double rhop = 0.05;
   const double U0 = 0.5;
   const double Up = 0.05;
   const double T0 = 1.0;
   const double Tp = 0.05;
   u.SetSize(4);
   u(0) = rho0 + rhop*pow(sin(M_PI*x(0)),2)*sin(M_PI*x(1));
   u(1) = 4.0*U0*x(1)*(1.0 - x(1)) + Up*sin(2 * M_PI * x(1)) * pow(sin(M_PI * x(0)),2);
   u(2) = -Up*pow(sin(2 * M_PI * x(0)),2) * sin(M_PI * x(1));
   double T = T0 + Tp*(pow(x(0), 4) - 2 * pow(x(0), 3) + pow(x(0), 2) 
                     + pow(x(1), 4) - 2 * pow(x(1), 3) + pow(x(1), 2));
   double p = u(0)*T; // T is nondimensionalized by 1/(R*a_infty^2)
   u(3) = p/euler::gami + 0.5*u(0)*(u(1)*u(1) + u(2)*u(2));
   u(1) *= u(0);
   u(2) *= u(0);
}

} // namespace mach 

