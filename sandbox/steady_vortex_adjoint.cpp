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
      unique_ptr<Mesh> smesh = buildQuarterAnnulusMesh(map_degree, nx, ny);
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
      double drag = abs(solver->calcOutput("drag") - (-1 / mach::euler::gamma));

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
