/// Solve the Navier-Stokes MMS verification

#include "mfem.hpp"
#include "navier_stokes.hpp"
#include <fstream>
#include <iostream>

using namespace std;
using namespace mfem;
using namespace mach;

/// Generate smoothly perturbed mesh 
/// \param[in] degree - polynomial degree of the mapping
/// \param[in] num_x - number of nodes in the x direction
/// \param[in] num_y - number of nodes in the y direction
std::unique_ptr<Mesh> buildCurvilinearMesh(int degree, int num_x, int num_y);

/// \brief Defines the exact solution for the manufactured solution
/// \param[in] x - coordinate of the point at which the state is needed
/// \param[out] u - state variables stored as a 4-vector
void uexact(const Vector &x, Vector& u);

int main(int argc, char *argv[])
{
   const char *options_file = "navier_stokes_mms_options.json";

   // Initialize MPI
   int num_procs, rank;
   MPI_Init(&argc, &argv);
   MPI_Comm_size(MPI_COMM_WORLD, &num_procs);
   MPI_Comm_rank(MPI_COMM_WORLD, &rank);
   ostream *out = getOutStream(rank);
   *out << std::setprecision(15); 

   // Parse command-line options
   OptionsParser args(argc, argv);

   int degree = 2.0;
   int nx = 20;
   int ny = 2;
   args.AddOption(&options_file, "-o", "--options",
                  "Options file to use.");
   args.AddOption(&degree, "-d", "--degree", "poly. degree of mesh mapping");
   args.AddOption(&nx, "-nx", "--num-x", "number of x-direction segments");
   args.AddOption(&ny, "-ny", "--num-y", "number of y-direction segments");
   args.Parse();
   if (!args.Good())
   {
      args.PrintUsage(*out);
      return 1;
   }

   try
   {
      // construct the mesh
      string opt_file_name(options_file);
      unique_ptr<Mesh> smesh = buildCurvilinearMesh(degree, nx, ny);
      *out << "Number of elements " << smesh->GetNE() <<'\n';
      ofstream sol_ofs("navier_stokes_mms_mesh.vtk");
      sol_ofs.precision(14);
      smesh->PrintVTK(sol_ofs, 3);

      // construct the solver and set the initial condition
      auto solver = createSolver<NavierStokesSolver<2>>(opt_file_name,
                                                        move(smesh));
      solver->setInitialCondition(uexact);
      solver->printSolution("init", degree+1);
      solver->printResidual("init-res", degree+1);

      *out << "\n|| rho_h - rho ||_{L^2} = " 
                << solver->calcL2Error(uexact, 0) << '\n' << endl;
      *out << "\ninitial residual norm = " << solver->calcResidualNorm() << endl;

      solver->solveForState();
      solver->printSolution("final", degree+1);
      double drag = solver->calcOutput("drag");

      *out << "\n|| rho_h - rho ||_{L^2} = " 
                << solver->calcL2Error(uexact, 0) << '\n' << endl;
      *out << "\ndrag \"error\" = " << drag - 1.6 << endl;
      *out << "\nfinal residual norm = " << solver->calcResidualNorm() << endl;

      // TEMP
      //static_cast<NavierStokesSolver<2>*>(solver.get())->setSolutionError(uexact);
      //solver->printSolution("error", degree+1);

      // Solve for and print out the adjoint
      solver->solveForAdjoint("drag");
      solver->printAdjoint("adjoint", degree+1);

   }
   catch (MachException &exception)
   {
      exception.print_message();
   }
   catch (std::exception &exception)
   {
      cerr << exception.what() << endl;
   }

   MPI_Finalize();
}

std::unique_ptr<Mesh> buildCurvilinearMesh(int degree, int num_x, int num_y)
{
   auto mesh_ptr = unique_ptr<Mesh>(new Mesh(num_x, num_y,
                                             Element::TRIANGLE, true /* gen. edges */,
                                             1.0, 1.0, true));
   // strategy:
   // 1) generate a fes for Lagrange elements of desired degree
   // 2) create a Grid Function using a VectorFunctionCoefficient
   // 4) use mesh_ptr->NewNodes(nodes, true) to set the mesh nodes
   
   // Problem: fes does not own fec, which is generated in this function's scope
   // Solution: the grid function can own both the fec and fes
   H1_FECollection *fec = new H1_FECollection(degree, 2 /* = dim */);
   FiniteElementSpace *fes = new FiniteElementSpace(mesh_ptr.get(), fec, 2,
                                                    Ordering::byVDIM);

   auto xy_fun = [](const Vector& xi, Vector &x)
   {
      x(0) = xi(0) + (1.0/40.0)*sin(2.0*M_PI*xi(0))*sin(2.0*M_PI*xi(1));
      x(1) = xi(1) + (1.0/40.0)*sin(2.0*M_PI*xi(1))*sin(2.0*M_PI*xi(0));
   };
   VectorFunctionCoefficient xy_coeff(2, xy_fun);
   GridFunction *xy = new GridFunction(fes);
   xy->MakeOwner(fec);
   xy->ProjectCoefficient(xy_coeff);
   mesh_ptr->NewNodes(*xy, true);
   return mesh_ptr;
}

void uexact(const Vector &x, Vector& q)
{
   const double rho0 = 1.0;
   const double rhop = 0.05;
   const double U0 = 0.5;
   const double Up = 0.05;
   const double T0 = 1.0;
   const double Tp = 0.05;
   q.SetSize(4);
   q(0) = rho0 + rhop*pow(sin(M_PI*x(0)),2)*sin(M_PI*x(1));
   q(1) = 4.0*U0*x(1)*(1.0 - x(1)) + Up*sin(2 * M_PI * x(1)) * pow(sin(M_PI * x(0)),2);
   q(2) = -Up*pow(sin(2 * M_PI * x(0)),2) * sin(M_PI * x(1));
   double T = T0 + Tp*(pow(x(0), 4) - 2 * pow(x(0), 3) + pow(x(0), 2) 
                     + pow(x(1), 4) - 2 * pow(x(1), 3) + pow(x(1), 2));
   double p = q(0)*T; // T is nondimensionalized by 1/(R*a_infty^2)
   q(3) = p/euler::gami + 0.5*q(0)*(q(1)*q(1) + q(2)*q(2));
   q(1) *= q(0);
   q(2) *= q(0);
}
