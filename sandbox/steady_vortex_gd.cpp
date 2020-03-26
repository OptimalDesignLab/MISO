/// Solve the steady isentropic vortex problem on a quarter annulus
#include<random>
#include "adept.h"

#include "mfem.hpp"
#include "euler.hpp"
#include "galer_diff.hpp"
#include "centgridfunc.hpp"
#include <fstream>
#include <iostream>

using namespace std;
using namespace mfem;
using namespace mach;

std::default_random_engine gen(std::random_device{}());
std::uniform_real_distribution<double> normal_rand(-1.0,1.0);

/// \brief Defines the exact solution for the steady isentropic vortex
/// \param[in] x - coordinate of the point at which the state is needed
/// \param[out] u - conservative variables stored as a 4-vector
void uexact(const Vector &x, Vector& u);

/// \brief Defines the random function for the jabocian check
/// \param[in] x - coordinate of the point at which the state is needed
/// \param[out] u - conservative variables stored as a 4-vector
void pert(const Vector &x, Vector& p);

/// Generate quarter annulus mesh 
/// \param[in] degree - polynomial degree of the mapping
/// \param[in] num_rad - number of nodes in the radial direction
/// \param[in] num_ang - number of nodes in the angular direction
std::unique_ptr<Mesh> buildQuarterAnnulusMesh(int degree, int num_rad,
                                              int num_ang);

int main(int argc, char *argv[])
{
   const char *options_file = "steady_vortex_gd_options.json";
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
#endif
#ifdef MFEM_USE_PETSC
   MFEMInitializePetsc(NULL, NULL, petscrc_file, NULL);
#endif
  
   // Parse command-line options
   OptionsParser args(argc, argv);
   int degree = 2.0;
   int nx = 1;
   int ny = 1;
   args.AddOption(&options_file, "-o", "--options",
                  "Options file to use.");
   args.AddOption(&degree, "-d", "--degree", "poly. degree of mesh mapping");
   args.AddOption(&nx, "-nr", "--num-rad", "number of radial segments");
   args.AddOption(&ny, "-nt", "--num-thetat", "number of angular segments");
   args.Parse();
   if (!args.Good())
   {
      args.PrintUsage(cout);
      return 1;
   }
  
   try
   {
      PCU_Comm_Init();
#ifdef MFEM_USE_SIMMETRIX
      Sim_readLicenseFile(0);
      gmi_sim_start();
      gmi_register_sim();
#endif
      gmi_register_mesh();
      // construct the solver, set the initial condition, and solve
      string opt_file_name(options_file);

      unique_ptr<AbstractSolver> solver(new EulerSolver<2>(opt_file_name, nullptr));
      solver->initDerived();
      std::cout << "initDerived is called.\n";

      solver->setInitialCondition(uexact);
      solver->printSolution("gd_serial_init", 0);
      std::cout << "initial solution is printed.\n";
      
      solver->setperturb(pert);
      solver->jacobianCheck();

      double l_error = solver->calcL2Error(uexact, 0);
      double res_error = solver->calcResidualNorm();
      if (0==myid)
      {
         mfem::out << "\n|| rho_h - rho ||_{L^2} = " << l_error;
         mfem::out << "\ninitial residual norm = " << res_error << endl;
      }

      solver->solveForState();
      solver->printSolution("gd_serial_final",0);

      l_error = solver->calcL2Error(uexact, 0);
      res_error = solver->calcResidualNorm();
      //double drag = abs(solver->calcOutput("drag") - (-1 / mach::euler::gamma));

      if (0==myid)
      {
         mfem::out << "\nfinal residual norm = " << res_error;
         mfem::out << "\n|| rho_h - rho ||_{L^2} = " << l_error << endl;
         //mfem::out << "\nDrag error = " << drag << endl;
      }
      PCU_Comm_Free();
#ifdef MFEM_USE_SIMMETRIX
      gmi_sim_stop();
      Sim_unregisterAllKeys();
#endif
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

// Exact solution; note that I reversed the flow direction to be clockwise, so
// the problem and mesh are consistent with the LPS paper (that is, because the
// triangles are subdivided from the quads using the opposite diagonal)
void uexact(const Vector &x, Vector& u)
{
   u.SetSize(4);
   double ri = 1.0;
   double Mai = 0.5; //0.95 
   double rhoi = 2.0;
   double prsi = 1.0/euler::gamma;
   double rinv = ri/sqrt((x(0)+1.0)*(x(0)+1.0) + (x(1)+1.0)*(x(1)+1.0));
   double rho = rhoi*pow(1.0 + 0.5*euler::gami*Mai*Mai*(1.0 - rinv*rinv),
                         1.0/euler::gami);
   double Ma = sqrt((2.0/euler::gami)*( ( pow(rhoi/rho, euler::gami) ) * 
                    (1.0 + 0.5*euler::gami*Mai*Mai) - 1.0 ) );
   double theta;
   if ((x(0)+1.0) > 1e-15)
   {
      theta = atan((x(1)+1.0)/(x(0)+1.0));
   }
   else
   {
      theta = M_PI/2.0;
   }
   double press = prsi* pow( (1.0 + 0.5*euler::gami*Mai*Mai) / 
                 (1.0 + 0.5*euler::gami*Ma*Ma), euler::gamma/euler::gami);
   double a = sqrt(euler::gamma*press/rho);

   u(0) = rho;
   u(1) = rho*a*Ma*sin(theta);
   u(2) = -rho*a*Ma*cos(theta);
   u(3) = press/euler::gami + 0.5*rho*a*a*Ma*Ma;
}

unique_ptr<Mesh> buildQuarterAnnulusMesh(int degree, int num_rad, int num_ang)
{
   auto mesh_ptr = unique_ptr<Mesh>(new Mesh(num_rad, num_ang,
                                             Element::TRIANGLE, true /* gen. edges */,
                                             2.0, M_PI*0.5, true));
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
   auto xy_fun = [](const Vector& rt, Vector &xy)
   {
      xy(0) = (rt(0) + 1.0)*cos(rt(1)); // need + 1.0 to shift r away from origin
      xy(1) = (rt(0) + 1.0)*sin(rt(1));
   };
   VectorFunctionCoefficient xy_coeff(2, xy_fun);
   GridFunction *xy = new GridFunction(fes);
   xy->MakeOwner(fec);
   xy->ProjectCoefficient(xy_coeff);

   mesh_ptr->NewNodes(*xy, true);
   return mesh_ptr;
}
