/// Solve the steady isentropic vortex problem on a quarter annulus
// set this const expression to true in order to use entropy variables for state
constexpr bool entvar = false;

#include<random>
#include "adept.h"
#include "mfem.hpp"
#include "euler.hpp"
#include <fstream>
#include <iostream>

using namespace std;
using namespace mfem;
using namespace mach;

std::default_random_engine gen(std::random_device{}());
std::uniform_real_distribution<double> normal_rand(-1.0,1.0);

/// \brief Defines the random function for the jabocian check
/// \param[in] x - coordinate of the point at which the state is needed
/// \param[out] u - conservative variables stored as a 4-vector
void pert(const Vector &x, Vector& p);

/// \brief Returns the value of the integrated math entropy over the domain
double calcEntropyTotalExact();

/// \brief Defines the exact solution for the steady isentropic vortex
/// \param[in] x - coordinate of the point at which the state is needed
/// \param[out] u - state variables stored as a 4-vector
void uexact(const Vector &x, Vector& u);

/// Generate quarter annulus mesh 
/// \param[in] degree - polynomial degree of the mapping
/// \param[in] num_rad - number of nodes in the radial direction
/// \param[in] num_ang - number of nodes in the angular direction
std::unique_ptr<Mesh> buildQuarterAnnulusMesh(int degree, int num_rad,
                                              int num_ang);

int main(int argc, char *argv[])
{
   const char *options_file = "euler_ellipse_options.json";
#ifdef MFEM_USE_PETSC
   const char *petscrc_file = "eulersteady.petsc";
   // Get the option file
   nlohmann::json options;
   ifstream option_source(options_file);
   option_source >> options;
   // Write the petsc option file
   ofstream petscoptions(petscrc_file);
   const string linearsolver_name = options["petscsolver"]["ksptype"].get<string>();
   const string prec_name = options["petscsolver"]["pctype"].get<string>();
   petscoptions << "-solver_ksp_type " << linearsolver_name << '\n';
   petscoptions << "-prec_pc_type " << prec_name << '\n';
   //petscoptions << "-prec_pc_factor_levels " << 4 << '\n';

   petscoptions.close();
#endif
#ifdef MFEM_USE_MPI
   // Initialize MPI if parallel
   int num_procs, myid;
   MPI_Init(&argc, &argv);
   MPI_Comm_size(MPI_COMM_WORLD, &num_procs);
   MPI_Comm_rank(MPI_COMM_WORLD, &myid);
#else
   int myid = 0;
#endif
#ifdef MFEM_USE_PETSC
   MFEMInitializePetsc(NULL, NULL, petscrc_file, NULL);
#endif
  
   // Parse command-line options
   OptionsParser args(argc, argv);
   int degree = 2;
   int nx = 1;
   int ny = 1;
   args.AddOption(&options_file, "-o", "--options",
                  "Options file to use.");
   args.AddOption(&degree, "-d", "--degree", "poly. degree of mesh mapping");
//    args.AddOption(&nx, "-nr", "--num-rad", "number of radial segments");
//    args.AddOption(&ny, "-nt", "--num-theta", "number of angular segments");
   args.Parse();
   if (!args.Good())
   {
      args.PrintUsage(cout);
      return 1;
   }
  
   try
   {
      // construct the solver, set the initial condition, and solve
      // string opt_file_name(options_file);
      // unique_ptr<Mesh> smesh = buildQuarterAnnulusMesh(degree, nx, ny);
      // ofstream sol_ofs("steady_vortex_mesh.vtk");
      // sol_ofs.precision(14);
      // smesh->PrintVTK(sol_ofs);
      // sol_ofs.close();
      string opt_file_name(options_file);
      unique_ptr<Mesh> smesh = buildQuarterAnnulusMesh(degree, nx, ny);
      std::cout << "Number of elements " << smesh->GetNE() <<'\n';
      ofstream sol_ofs("steady_vortex_mesh.vtk");
      ofstream meshsave("steady_vortex_mesh.mesh");
      sol_ofs.precision(14);
      smesh->PrintVTK(sol_ofs,0);
      smesh->Print(meshsave);
      sol_ofs.close();
      meshsave.close();

      unique_ptr<AbstractSolver> solver(
         new EulerSolver<2, entvar>(opt_file_name, move(smesh)));
      //unique_ptr<AbstractSolver> solver(new EulerSolver<2>(opt_file_name, nullptr));
      solver->initDerived();

      solver->setInitialCondition(uexact);
      solver->printSolution("gd_init", 0);

      // get the initial density error
      double l2_error = (static_cast<EulerSolver<2, entvar>&>(*solver)
                            .calcConservativeVarsL2Error(uexact, 0));
      double res_error = solver->calcResidualNorm();
      if (0==myid)
      {
         mfem::out << "\n|| rho_h - rho ||_{L^2} = " << l2_error;
         mfem::out << "\ninitial residual norm = " << res_error << endl;
      }
      solver->checkJacobian(pert);
      solver->feedpert(pert);
      solver->solveForState();
      solver->printSolution("gd_final",0);
      solver->printError("gd_final_error", 0, uexact);
      // get the final density error
      l2_error = (static_cast<EulerSolver<2, entvar>&>(*solver)
                            .calcConservativeVarsL2Error(uexact, 0));
      res_error = solver->calcResidualNorm();
      double drag = abs(solver->calcOutput("drag"));
      double entropy = solver->calcOutput("entropy");

      if (0==myid)
      {
         mfem::out << "\nfinal residual norm = " << res_error;
        //  mfem::out << "\n|| rho_h - rho ||_{L^2} = " << l2_error << endl;
         mfem::out << "\nDrag error = " << drag << endl;
        //  mfem::out << "\nTotal entropy = " << entropy;
        //  mfem::out << "\nEntropy error = "
        //            << fabs(entropy - calcEntropyTotalExact()) << endl;
      }

   }
   catch (MachException &exception)
   {
      exception.print_message();
   }
   catch (std::exception &exception)
   {
      cerr << exception.what() << endl;
   }

#ifdef MFEM_USE_PETSC
   MFEMFinalizePetsc();
#endif
#ifdef MFEM_USE_MPI
   MPI_Finalize();
#endif
}

// perturbation function used to check the jacobian in each iteration
void pert(const Vector &x, Vector& p)
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
   double prsi = 1.0/euler::gamma;
   double si = log(prsi/pow(rhoi, euler::gamma));
   return -si*8.746553803443305*M_PI*0.5/0.4;
}

// Exact solution; note that I reversed the flow direction to be clockwise, so
// the problem and mesh are consistent with the LPS paper (that is, because the
// triangles are subdivided from the quads using the opposite diagonal)
void uexact(const Vector &x, Vector& q)
{
   q.SetSize(4);
   double mach_fs = 0.5;
   q(0) = 1.0;
   q(1) = q(0) * mach_fs; // ignore angle of attack
   q(2) = 0.0;
   q(3) = 1 / (euler::gamma * euler::gami) + 0.5 * mach_fs * mach_fs;
}

/// use this for flow over an ellipse
unique_ptr<Mesh> buildQuarterAnnulusMesh(int degree, int num_rad, int num_ang)
{
   int ref_levels = 5;
   const char *mesh_file = "periodic_rectangle_tri.mesh";
   auto mesh_ptr = unique_ptr<Mesh>(new Mesh(mesh_file, 1, 1));
   for (int l = 0; l < ref_levels; l++)
   {
      mesh_ptr->UniformRefinement();
   }
   H1_FECollection *fec = new H1_FECollection(degree, 2 /* = dim */);
   FiniteElementSpace *fes = new FiniteElementSpace(mesh_ptr.get(), fec, 2,
                                                    Ordering::byVDIM);

   // This lambda function transforms from (r,\theta) space to (x,y) space
   auto xy_fun = [](const Vector &rt, Vector &xy) {
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