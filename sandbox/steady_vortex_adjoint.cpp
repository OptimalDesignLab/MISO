/// Solve the steady isentropic vortex problem on a quarter annulus, and then
/// solves the adjoint corresponding to the drag computed over the inner radius
#include<random>
#include "adept.h"

#include "mfem.hpp"
#include "euler.hpp"
#include <fstream>
#include <iostream>

using namespace std;
using namespace mfem;
using namespace miso;

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

int main(int argc, char *argv[])
{
   const char *options_file = "steady_vortex_adjoint_options.json";

   // Initialize MPI
   int num_procs, myid;
   MPI_Init(&argc, &argv);
   MPI_Comm_size(MPI_COMM_WORLD, &num_procs);
   MPI_Comm_rank(MPI_COMM_WORLD, &myid);
  
   // Parse command-line options
   OptionsParser args(argc, argv);
   int map_degree = 2;
   int nx = 1;
   int ny = 1;
   args.AddOption(&options_file, "-o", "--options",
                  "Options file to use.");
   args.AddOption(&map_degree, "-d", "--degree", "poly. degree of mesh mapping");
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
      // construct the mesh 
      string opt_file_name(options_file);
      auto smesh = buildQuarterAnnulusMesh(map_degree, nx, ny);
      std::cout << "Number of elements " << smesh->GetNE() << std::endl;
      ofstream sol_ofs("steady_vortex_adjoint_mesh.vtk");
      sol_ofs.precision(14);
      smesh->PrintVTK(sol_ofs,3);

      // construct the solver and set initial conditions
      auto solver = createSolver<EulerSolver<2> >(opt_file_name, move(smesh));
      solver->setInitialCondition(uexact);
      solver->printSolution("init", map_degree+1);

      double l2_error = solver->calcL2Error(uexact, 0);
      double res_error = solver->calcResidualNorm();
      if (0==myid)
      {
         mfem::out << "\n|| rho_h - rho ||_{L^2} = " << l2_error;
         mfem::out << "\ninitial residual norm = " << res_error << endl;
      }
      solver->checkJacobian(pert);
      solver->solveForState();
      solver->printSolution("final", map_degree+1);
      l2_error = solver->calcL2Error(uexact, 0);
      res_error = solver->calcResidualNorm();
      double drag = abs(solver->calcOutput("drag") - (-1 / miso::euler::gamma));

      if (0==myid)
      {
         mfem::out << "\nfinal residual norm = " << res_error;
         mfem::out << "\n|| rho_h - rho ||_{L^2} = " << l2_error << endl;
         mfem::out << "\nDrag error = " << drag << endl;
      }

      // Solve for and print out the adjoint
      solver->solveForAdjoint("drag");
      solver->printAdjoint("adjoint", map_degree+1);

   }
   catch (MISOException &exception)
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

   MPI_Finalize();
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
   double rinv = ri/sqrt(x(0)*x(0) + x(1)*x(1));
   double rho = rhoi*pow(1.0 + 0.5*euler::gami*Mai*Mai*(1.0 - rinv*rinv),
                         1.0/euler::gami);
   double Ma = sqrt((2.0/euler::gami)*( ( pow(rhoi/rho, euler::gami) ) * 
                    (1.0 + 0.5*euler::gami*Mai*Mai) - 1.0 ) );
   double theta;
   if (x(0) > 1e-15)
   {
      theta = atan(x(1)/x(0));
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
