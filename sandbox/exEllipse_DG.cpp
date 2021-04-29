/// Solve the steady isentropic vortex problem on a quarter annulus

// set this const expression to true in order to use entropy variables for state
constexpr bool entvar = false;
#include "mfem.hpp"
#include <fstream>
#include <iostream>
#include <random>
#include "euler_integ_DG.hpp"
#include "evolver.hpp"
#include "json.hpp"
#include "default_options.hpp"
#include "mfem_extensions.hpp"
using namespace std;
using namespace mfem;
using namespace mach;
std::default_random_engine gen(std::random_device{}());
std::uniform_real_distribution<double> normal_rand(-1.0, 1.0);
static std::uniform_real_distribution<double> uniform_rand(0.0, 1.0);
const double rho = 0.9856566615165173;
const double rhoe = 2.061597236955558;
const double rhou[3] = {0.09595562550099601, -0.030658751626551423, -0.13471469906596886};

unique_ptr<Solver> constructLinearSolver(
    nlohmann::json &_options, mfem::Solver &_prec, MPI_Comm comm)
{
   std::string solver_type = _options["type"].get<std::string>();
   double reltol = _options["reltol"].get<double>();
   int maxiter = _options["maxiter"].get<int>();
   int ptl = _options["printlevel"].get<int>();
   int kdim = _options.value("kdim", -1);

   unique_ptr<Solver> lin_solver;
   if (solver_type == "hypregmres")
   {
      lin_solver.reset(new HypreGMRES(comm));
      HypreGMRES *gmres = dynamic_cast<HypreGMRES *>(lin_solver.get());
      gmres->SetTol(reltol);
      gmres->SetMaxIter(maxiter);
      gmres->SetPrintLevel(ptl);
      gmres->SetPreconditioner(dynamic_cast<HypreSolver &>(_prec));
      if (kdim != -1)
         gmres->SetKDim(kdim); // set GMRES subspace size
   }
   else if (solver_type == "hyprefgmres")
   {
      lin_solver.reset(new HypreFGMRES(comm));
      HypreFGMRES *fgmres = dynamic_cast<HypreFGMRES *>(lin_solver.get());
      fgmres->SetTol(reltol);
      fgmres->SetMaxIter(maxiter);
      fgmres->SetPrintLevel(ptl);
      fgmres->SetPreconditioner(dynamic_cast<HypreSolver &>(_prec));
      if (kdim != -1)
         fgmres->SetKDim(kdim); // set FGMRES subspace size
   }
   else if (solver_type == "gmressolver")
   {
      lin_solver.reset(new GMRESSolver(comm));
      GMRESSolver *gmres = dynamic_cast<GMRESSolver *>(lin_solver.get());
      gmres->SetRelTol(reltol);
      gmres->SetMaxIter(maxiter);
      gmres->SetPrintLevel(ptl);
      gmres->SetPreconditioner(dynamic_cast<Solver &>(_prec));
      if (kdim != -1)
         gmres->SetKDim(kdim); // set GMRES subspace size
   }
   else if (solver_type == "hyprepcg")
   {
      lin_solver.reset(new HyprePCG(comm));
      HyprePCG *pcg = static_cast<HyprePCG *>(lin_solver.get());
      pcg->SetTol(reltol);
      pcg->SetMaxIter(maxiter);
      pcg->SetPrintLevel(ptl);
      pcg->SetPreconditioner(dynamic_cast<HypreSolver &>(_prec));
   }
   else if (solver_type == "cgsolver")
   {
      lin_solver.reset(new CGSolver(comm));
      CGSolver *cg = dynamic_cast<CGSolver *>(lin_solver.get());
      cg->SetRelTol(reltol);
      cg->SetMaxIter(maxiter);
      cg->SetPrintLevel(ptl);
      cg->SetPreconditioner(dynamic_cast<Solver &>(_prec));
   }
   else
   {
      throw MachException("Unsupported iterative solver type!\n"
                          "\tavilable options are: HypreGMRES, HypreFGMRES, GMRESSolver,\n"
                          "\tHyprePCG, CGSolver");
   }
   cout << "Linear solver " << solver_type << " is set " << endl;
   return lin_solver;
}

unique_ptr<NewtonSolver> constructNonlinearSolver(
    nlohmann::json &_options, mfem::Solver &_lin_solver, MPI_Comm comm)
{
   std::string solver_type = _options["type"].get<std::string>();
   double abstol = _options["abstol"].get<double>();
   double reltol = _options["reltol"].get<double>();
   int maxiter = _options["maxiter"].get<int>();
   int ptl = _options["printlevel"].get<int>();
   unique_ptr<NewtonSolver> nonlin_solver;
   if (solver_type == "newton")
   {
      nonlin_solver.reset(new mfem::NewtonSolver(comm));
   }
   else
   {
      throw MachException("Unsupported nonlinear solver type!\n"
                          "\tavilable options are: newton\n");
   }
   //double eta = 1e-1;
   //newton_solver.reset(new InexactNewton(comm, eta));

   nonlin_solver->iterative_mode = true;
   nonlin_solver->SetSolver(dynamic_cast<Solver &>(_lin_solver));
   nonlin_solver->SetPrintLevel(ptl);
   nonlin_solver->SetRelTol(reltol);
   nonlin_solver->SetAbsTol(abstol);
   nonlin_solver->SetMaxIter(maxiter);
   cout << solver_type << " solver is set " << endl;
   return nonlin_solver;
}

unique_ptr<Solver> constructPreconditioner(
    nlohmann::json &_options, mach::SpaceType *fes, MPI_Comm comm)
{
   std::string prec_type = _options["type"].get<std::string>();
   unique_ptr<Solver> precond;
   if (prec_type == "hypreeuclid")
   {
      precond.reset(new HypreEuclid(comm));
      // TODO: need to add HYPRE_EuclidSetLevel to odl branch of mfem
      cout << "WARNING! Euclid fill level is hard-coded"
           << "(see AbstractSolver::constructLinearSolver() for details)" << endl;
      //int fill = options["lin-solver"]["filllevel"].get<int>();
      //HYPRE_EuclidSetLevel(dynamic_cast<HypreEuclid*>(precond.get())->GetPrec(), fill);
   }
   else if (prec_type == "hypreilu")
   {
      precond.reset(new HypreILU());
      HypreILU *ilu = dynamic_cast<HypreILU *>(precond.get());
      HYPRE_ILUSetType(*ilu, _options["ilu-type"].get<int>());
      HYPRE_ILUSetLevelOfFill(*ilu, _options["lev-fill"].get<int>());
      HYPRE_ILUSetLocalReordering(*ilu, _options["ilu-reorder"].get<int>());
      HYPRE_ILUSetPrintLevel(*ilu, _options["printlevel"].get<int>());
// Just listing the options below in case we need them in the future
#if 0
      HYPRE_ILUSetSchurMaxIter(ilu, schur_max_iter);
      HYPRE_ILUSetNSHDropThreshold(ilu, nsh_thres); needs type = 20,21
      HYPRE_ILUSetDropThreshold(ilu, drop_thres);
      HYPRE_ILUSetMaxNnzPerRow(ilu, nz_max);
#endif
   }
   else if (prec_type == "hypreams")
   {
      precond.reset(new HypreAMS(fes));
      HypreAMS *ams = dynamic_cast<HypreAMS *>(precond.get());
      ams->SetPrintLevel(_options["printlevel"].get<int>());
      ams->SetSingularProblem();
   }
   else if (prec_type == "hypreboomeramg")
   {
      precond.reset(new HypreBoomerAMG());
      HypreBoomerAMG *amg = dynamic_cast<HypreBoomerAMG *>(precond.get());
      amg->SetPrintLevel(_options["printlevel"].get<int>());
   }
   else if (prec_type == "blockilu")
   {
      precond.reset(new BlockILU(4));
   }
   else
   {
      throw MachException("Unsupported preconditioner type!\n"
                          "\tavilable options are: HypreEuclid, HypreILU, HypreAMS,"
                          " HypreBoomerAMG.\n");
   }
   cout << prec_type << " Preconditioner is set " << endl;
   return precond;
}

template <int dim>
void randBaselinePert(const mfem::Vector &x, mfem::Vector &u)
{

   const double scale = 0.01;
   u(0) = rho * (1.0 + scale * uniform_rand(gen));
   u(dim + 1) = rhoe * (1.0 + scale * uniform_rand(gen));
   for (int di = 0; di < dim; ++di)
   {
      u(di + 1) = rhou[di] * (1.0 + scale * uniform_rand(gen));
   }
}

double calcDrag(mach::SpaceType *fes, mfem::GridFunction u,
                int num_state, double alpha)
{
     /// check initial drag value
   mfem::Vector drag_dir(2);

   drag_dir = 0.0;
   int iroll = 0;
   int ipitch = 1;
   double aoa_fs = 0.0;
   double mach_fs = 0.5;

   drag_dir(iroll) = cos(aoa_fs);
   drag_dir(ipitch) = sin(aoa_fs);
   drag_dir *= 1.0 / pow(mach_fs, 2.0); // to get non-dimensional Cd

   Array<int> bndry_marker_drag;
   bndry_marker_drag.Append(1);
   bndry_marker_drag.Append(0);
   NonlinearForm *dragf = new NonlinearForm(fes);

   dragf->AddBdrFaceIntegrator(
       new DG_PressureForce<2, entvar>(drag_dir, num_state, alpha),
       bndry_marker_drag);

   double drag = dragf->GetEnergy(u);
   return drag;
}

/// function to calculate conservative variables l2error
template <int dim, bool entvar>
double calcConservativeVarsL2Error(
    void (*u_exact)(const mfem::Vector &, mfem::Vector &), mach::GridFunType *u, mach::SpaceType *fes,
    int num_state, int entry)
{
   // This lambda function computes the error at a node
   // Beware: this is not particularly efficient, given the conditionals
   // Also **NOT thread safe!**
   Vector qdiscrete(dim + 2), qexact(dim + 2); // define here to avoid reallocation
   auto node_error = [&](const Vector &discrete, const Vector &exact) -> double {
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
   DenseMatrix vals, exact_vals;
   Vector u_j, exsol_j;
   double loc_norm = 0.0;
   for (int i = 0; i < fes->GetNE(); i++)
   {
      const FiniteElement *fe = fes->GetFE(i);
      const IntegrationRule *ir;

      int intorder = 2 * fe->GetOrder() + 3;
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
      }
   }

   double norm = loc_norm;
   if (norm < 0.0) // This was copied from mfem...should not happen for us
   {
      return -sqrt(-norm);
   }
   return sqrt(norm);
}

void randState(const mfem::Vector &x, mfem::Vector &u)
{
   for (int i = 0; i < u.Size(); ++i)
   {
      u(i) = 2.0 * uniform_rand(gen) - 1.0;
   }
}

double calcResidualNorm(mach::NonlinearFormType *res, mach::SpaceType *fes, ParGridFunction &u)
{
   ParGridFunction residual(fes);
   auto *u_true = u.GetTrueDofs();
   res->Mult(*u_true, residual);
   return residual.Norml2();
}

/// get freestream state values for the far-field bcs
template <int dim, bool entvar>
void getFreeStreamState(mfem::Vector &q_ref)
{
   double mach_fs = 0.5;
   q_ref = 0.0;
   q_ref(0) = 1.0;
   q_ref(1) = q_ref(0) * mach_fs; // ignore angle of attack
   q_ref(2) = 0.0;
   q_ref(dim + 1) = 1 / (euler::gamma * euler::gami) + 0.5 * mach_fs * mach_fs;
}

/// \brief Defines the random function for the jabocian check
/// \param[in] x - coordinate of the point at which the state is needed
/// \param[out] u - conservative variables stored as a 4-vector
void pert(const Vector &x, Vector &p);

/// \brief Returns the value of the integrated math entropy over the domain
double calcEntropyTotalExact();

/// \brief Defines the exact solution for the steady isentropic vortex
/// \param[in] x - coordinate of the point at which the state is needed
/// \param[out] u - state variables stored as a 4-vector
void uexact(const Vector &x, Vector &u);

/// Generate quarter annulus mesh
/// \param[in] degree - polynomial degree of the mapping
/// \param[in] num_rad - number of nodes in the radial direction
/// \param[in] num_ang - number of nodes in the angular direction
std::unique_ptr<Mesh> buildQuarterAnnulusMesh(int degree, int num_rad,
                                              int num_ang);

// main
int main(int argc, char *argv[])
{
   // 1. Initialize MPI.
   int num_procs, rank;
   MPI_Init(&argc, &argv);
   MPI_Comm_size(MPI_COMM_WORLD, &num_procs);
   MPI_Comm_rank(MPI_COMM_WORLD, &rank);
   ostream *out = getOutStream(rank);
   // Parse command-line options
   OptionsParser args(argc, argv);
   int degree = 2;
   int nx = 5;
   int ny = 5;
   int order = 1;
   int ref_levels = -1;
   int nc_ref = -1;
   args.AddOption(&degree, "-d", "--degree", "poly. degree of mesh mapping");
   args.AddOption(&order, "-o", "--order",
                  "Finite element order (polynomial degree) >= 0.");
   args.AddOption(&nx, "-nr", "--num-rad", "number of radial segments");
   args.AddOption(&ny, "-nt", "--num-theta", "number of angular segments");
   args.AddOption(&ref_levels, "-ref", "--refine",
                  "refine levels");

   args.Parse();
   if (!args.Good())
   {
      if (rank == 0)
      {
         args.PrintUsage(cout);
      }
      MPI_Finalize();
      return 1;
   }
   if (rank == 0)
   {
      args.PrintOptions(cout);
   }

   /// solver options
   const char *options_file = "exEllipse_DG_options.json";
   string opt_file_name(options_file);
   nlohmann::json file_options;
   std::ifstream options_file1(opt_file_name);
   options_file1 >> file_options;
   nlohmann::json options;
   options = file_options;
   options = default_options;
   options.merge_patch(file_options);
   static adept::Stack diff_stack;

   /// degree = p+1
   degree = order + 1;
   /// number of state variables
   int num_state = 4;

   /// construct the mesh
   unique_ptr<Mesh> mesh = buildQuarterAnnulusMesh(degree, nx, ny);
   cout << "Number of elements " << mesh->GetNE() << '\n';

   /// dimension
   const int dim = mesh->Dimension();

   for (int l = 0; l < ref_levels; l++)
   {
      mesh->UniformRefinement();
   }

   cout << "Number of elements after refinement " << mesh->GetNE() << '\n';
   // save the initial mesh
   ofstream sol_ofs("steady_vortex_mesh_gd_nc.vtk");
   sol_ofs.precision(14);
   mesh->PrintVTK(sol_ofs, 1);
   ParMesh *pmesh = new ParMesh(MPI_COMM_WORLD, *mesh.get());
   // finite element collection
   FiniteElementCollection *fec = new DG_FECollection(order, dim);

   // finite element space
   ParFiniteElementSpace *fes = new ParFiniteElementSpace(pmesh, fec, num_state,
                                                          Ordering::byVDIM);


   HYPRE_Int glob_size = fes->GlobalTrueVSize();
   cout << "Number of unknowns: " << glob_size << endl;

    /// `bndry_marker_*` lists the boundaries associated with a particular BC
   Array<int> bndry_marker_slipwall;
   Array<int> bndry_marker_farfield;

   bndry_marker_farfield.Append(0);
   bndry_marker_farfield.Append(1);

   bndry_marker_slipwall.Append(1);
   bndry_marker_slipwall.Append(0);

   Vector qfs(dim + 2);

   getFreeStreamState<2, 0>(qfs);
   double alpha = 1.0;

   /// nonlinearform
   ParNonlinearForm *res = new ParNonlinearForm(fes);
   res->AddDomainIntegrator(new EulerDomainIntegrator<2>(diff_stack, num_state, alpha));
   res->AddBdrFaceIntegrator(new EulerBoundaryIntegrator<2, 2, 0>(diff_stack, fec, num_state, qfs, alpha),
                             bndry_marker_slipwall);
   res->AddBdrFaceIntegrator(new EulerBoundaryIntegrator<2, 3, 0>(diff_stack, fec, num_state, qfs, alpha),
                             bndry_marker_farfield);
   res->AddInteriorFaceIntegrator(new EulerFaceIntegrator<2>(diff_stack, fec, 1.0, num_state, alpha));

   /// check if the integrators are correct
   double delta = 1e-5;

   // initialize state; here we randomly perturb a constant state
   ParGridFunction q(fes);
   VectorFunctionCoefficient pert(num_state, randBaselinePert<2>);
   q.ProjectCoefficient(pert);
   // initialize the vector that the Jacobian multiplies
   ParGridFunction v(fes);
   VectorFunctionCoefficient v_rand(num_state, randState);
   v.ProjectCoefficient(v_rand);
   // evaluate the Jacobian and compute its product with v
   Operator &Jac = res->GetGradient(q);
   ParGridFunction jac_v(fes);
   Jac.Mult(v, jac_v);
   // now compute the finite-difference approximation...
   ParGridFunction q_pert(q), r(fes), jac_v_fd(fes);
   q_pert.Add(-delta, v);
   res->Mult(q_pert, r);
   q_pert.Add(2.0 * delta, v);
   res->Mult(q_pert, jac_v_fd);
   jac_v_fd -= r;
   jac_v_fd /= (2.0 * delta);

   for (int i = 0; i < jac_v.Size(); ++i)
   {
      //std::cout << std::abs(jac_v(i) - (jac_v_fd(i))) << "\n";
      MFEM_ASSERT(abs(jac_v(i) - (jac_v_fd(i))) <= 1e-08, "jacobian is incorrect");
   }

   /// bilinear form
   ParBilinearForm *mass = new ParBilinearForm(fes);
   mass->AddDomainIntegrator(new EulerMassIntegrator(num_state));
   mass->Assemble();
   mass->Finalize();


   /// grid function
   ParGridFunction u(fes);
   VectorFunctionCoefficient u0(num_state, uexact);
   u.ProjectCoefficient(u0);


   /// newton solver for the steady problem
   std::unique_ptr<mfem::NewtonSolver> newton_solver;
   /// linear system solver used in newton solver
   std::unique_ptr<mfem::Solver> solver;
   /// linear system preconditioner for solver in newton solver and adjoint
   std::unique_ptr<mfem::Solver> prec;

   /// time-marching method
   std::unique_ptr<mfem::ODESolver> ode_solver;
   ode_solver = NULL;
   *out << "ode-solver type = "
        << options["time-dis"]["ode-solver"].template get<string>() << endl;
   ode_solver.reset(new PseudoTransientSolver(out));
   cout << "ode_solver is set " << endl;
   prec = constructPreconditioner(options["lin-prec"], fes, fes->GetComm());
   solver = constructLinearSolver(options["lin-solver"], *prec, fes->GetComm());
   newton_solver = constructNonlinearSolver(options["nonlin-solver"], *solver, fes->GetComm());
   *out << "No essential BCs" << endl;
   /// Array that marks boundaries as essential
   mfem::Array<int> ess_bdr;
   if (mesh->bdr_attributes) // some meshes may not have boundary attributes
   {
      ess_bdr.SetSize(mesh->bdr_attributes.Max());
      ess_bdr = 0;
   }
   /// TimeDependentOperator
   unique_ptr<mach::MachEvolver> evolver(new MachEvolver(ess_bdr, NULL, mass,
                                                         res, NULL, NULL, NULL,
                                                         *out, 0.0,
                                                         TimeDependentOperator::Type::IMPLICIT));
   evolver->SetNewtonSolver(newton_solver.get());

   /// set up the evolver
   auto t = 0.0;
   evolver->SetTime(t);
   ode_solver->Init(*evolver);

   // solve the ode problem
   double res_norm0 = calcResidualNorm(res, fes, u);
   std::cout << "initial residual norm: " << res_norm0 << "\n";

   /// initial l2_err
   double l2_err_init = calcConservativeVarsL2Error<2, 0>(uexact, &u, fes,
                                                          num_state, 0);
   cout << "l2_err_init " << l2_err_init << endl;

   double res_norm;
   //int exponent = options["time-dis"]["res-exp"].template get<double>();
   int exponent = 2;
   double t_final = options["time-dis"]["t-final"].template get<double>();
   *out << "t_final is " << t_final << '\n';
   int ti;
   bool done = false;
   double dt = 0.0;
   double dt_init = options["time-dis"]["dt"].template get<double>();
   double dt_old;
   for (ti = 0; ti < options["time-dis"]["max-iter"].get<int>(); ++ti)
   {
      /// calculate timestep
      res_norm = calcResidualNorm(res, fes, u);
      dt_old = dt;
      cout << "res_norm " << res_norm << endl;
      cout << "rel res_norm " << res_norm / res_norm0  << endl;
      dt = dt_init * pow(res_norm0 / res_norm, exponent);
      dt = max(dt, dt_old);
      /// print iterations
      std::cout << "iter " << ti << ": time = " << t << ": dt = " << dt << endl;
      //   if (!options["time-dis"]["steady"].get<bool>())
      //      *out << " (" << round(100 * t / t_final) << "% complete)";
      //   *out << endl;
      HypreParVector *u_true = u.GetTrueDofs();
      if (res_norm <= 1e-11)
        break;

      if (isnan(res_norm))
         break;

      ode_solver->Step(*u_true, t, dt);
   }

   cout << "=========================================" << endl;
   std::cout << "final residual norm: " << res_norm << "\n";
   double drag = calcDrag(fes, u, num_state, alpha);
   double drag_err = abs(drag);
   cout << "drag: " << drag << endl;
   cout << "drag_error: " << drag_err << endl;
   ofstream finalsol_ofs("final_sol_vortex_parGD.vtk");
   finalsol_ofs.precision(14);
   mesh->PrintVTK(finalsol_ofs, 1);
   u.SaveVTK(finalsol_ofs, "Solution", 1);
   finalsol_ofs.close();

   /// calculate final solution error
//    double l2_err_rho = calcConservativeVarsL2Error<2, 0>(uexact, &u, fes,
//                                                          num_state, 0);
//    cout << "|| rho_h - rho ||_{L^2} = " << l2_err_rho << endl;
   cout << "=========================================" << endl;
   delete pmesh;
   delete fes;
   delete fec;
   MPI_Finalize();
   return 0;
}

// perturbation function used to check the jacobian in each iteration
void pert(const Vector &x, Vector &p)
{
   p.SetSize(4);
   for (int i = 0; i < 4; i++)
   {
      p(i) = normal_rand(gen);
   }
}

// Returns the exact total entropy value over the quarter annulus
// Note: the number 8.74655... that appears below is the integral of r*rho over the radii
// from 1 to 3.  It was approixmated using a degree 51 Gaussian quadrature.
double calcEntropyTotalExact()
{
   double rhoi = 2.0;
   double prsi = 1.0 / euler::gamma;
   double si = log(prsi / pow(rhoi, euler::gamma));
   return -si * 8.746553803443305 * M_PI * 0.5 / 0.4;
}

// Exact solution; note that I reversed the flow direction to be clockwise, so
// the problem and mesh are consistent with the LPS paper (that is, because the
// triangles are subdivided from the quads using the opposite diagonal)
void uexact(const Vector &x, Vector &q)
{
   q.SetSize(4);
   double mach_fs = 0.5;
   q(0) = 1.0;
   q(1) = q(0) * mach_fs; // ignore angle of attack
   q(2) = 0.0;
   q(3) = 1 / (euler::gamma * euler::gami) + 0.5 * mach_fs * mach_fs;
}

unique_ptr<Mesh> buildQuarterAnnulusMesh(int degree, int num_rad, int num_ang)
{
   int ref_levels = 3;
   const char *mesh_file = "periodic_rectangle_2.mesh";
   //const char *mesh_file = "periodic_rectangle_tri.mesh";
   auto mesh_ptr = unique_ptr<Mesh>(new Mesh(mesh_file, 1, 1));

   for (int l = 0; l < ref_levels; l++)
   {
      mesh_ptr->UniformRefinement();
   }
   cout << "Number of elements " << mesh_ptr->GetNE() << '\n';
   // strategy:
   // 1) generate a fes for Lagrange elements of desired degree
   // 2) create a Grid Function using a VectorFunctionCoefficient
   // 4) use mesh_ptr->NewNodes(nodes, true) to set the mesh nodes

   // Problem: fes does not own fec, which is generated in this function's scope
   // Solution: the grid function can own both the fec and fes
   H1_FECollection *fec = new H1_FECollection(degree, 2 /* = dim */);
   FiniteElementSpace *fes = new FiniteElementSpace(mesh_ptr.get(), fec, 2,
                                                    Ordering::byVDIM);

  
   // This lambda function transforms from (r,\theta) space to (x,y) space
   auto xy_fun = [](const Vector &rt, Vector &xy) {
     
      // double r_far = 20.0;
      // double a0 = 0.5;
      // double b0 = a0 / 10.0;
      // double delta = 3.00; // We will have to experiment with this
      // double r = 1.0 + tanh(delta * (rt(0) / r_far - 1.0)) / tanh(delta);
      // double theta = rt(1);
      // double b = b0 + (a0 - b0) * r;
      // xy(0) = a0 * (r * r_far + 1.0) * cos(theta) + 10.0;
      // xy(1) = b * (r * r_far + 1.0) * sin(theta) + 10.0;
      /// using conformal mapping 
      double r_far = 60.0;
      double r = rt(0);
      double theta = rt(1);
      double ratio = 10.0;
      double delta = 3.0; // We will have to experiment with this
      double rf = 1.0 + tanh(delta * (rt(0) / r_far - 1.0)) / tanh(delta);
      double a = sqrt((1 + ratio) / (ratio - 1));
      xy(0) = a * (rf * r_far + 1) * cos(theta); // need +a to shift r away from origin
      xy(1) = a * (rf * r_far + 1) * sin(theta);
      /// using conformal mapping
      double rs = sqrt((xy(0) * xy(0)) + (xy(1) * xy(1)));
      double ax = (rs + 1.0 / rs);
      double ay = (rs - 1.0 / rs);
      xy(0) = (ax * cos(theta)) / 4.0 + 20.0;
      xy(1) = (ay * sin(theta)) / 4.0 + 20.0;
   };
   VectorFunctionCoefficient xy_coeff(2, xy_fun);
   GridFunction *xy = new GridFunction(fes);
   xy->MakeOwner(fec);
   xy->ProjectCoefficient(xy_coeff);

   mesh_ptr->NewNodes(*xy, true);
   return mesh_ptr;
}