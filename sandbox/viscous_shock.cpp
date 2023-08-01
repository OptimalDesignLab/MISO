/// Solve the viscous shock verification problem

#include "mfem.hpp"
#include "navier_stokes.hpp"
#include <fstream>
#include <iostream>

using namespace std;
using namespace mfem;
using namespace miso;

/// Generate smoothly perturbed mesh 
/// \param[in] degree - polynomial degree of the mapping
/// \param[in] num_x - number of nodes in the x direction
/// \param[in] num_y - number of nodes in the y direction
Mesh buildCurvilinearMesh(int degree, int num_x, int num_y);

int main(int argc, char *argv[])
{
   const char *options_file = "viscous_shock_options.json";

   // Initialize MPI
   int num_procs, rank;
   MPI_Init(&argc, &argv);
   MPI_Comm_size(MPI_COMM_WORLD, &num_procs);
   MPI_Comm_rank(MPI_COMM_WORLD, &rank);
   ostream *out = getOutStream(rank);

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
      unique_ptr<Mesh> smesh(new Mesh(buildCurvilinearMesh(degree, nx, ny)));
      *out << "Number of elements " << smesh->GetNE() <<'\n';
      ofstream sol_ofs("viscous_shock_mesh.vtk");
      sol_ofs.precision(14);
      smesh->PrintVTK(sol_ofs, 3);

      // construct the solver and set the initial condition
      auto solver = createSolver<NavierStokesSolver<2>>(opt_file_name,
                                                        move(smesh));
      solver->setInitialCondition(shockExact);
      solver->printSolution("init", degree+1);
      solver->printResidual("init-res", degree+1);

      *out << "\n|| rho_h - rho ||_{L^2} = " 
                << solver->calcL2Error(shockExact, 0) << '\n' << endl;
      *out << "\ninitial residual norm = " << solver->calcResidualNorm() << endl;

      solver->solveForState();
      solver->printSolution("final", degree+1);

      *out << "\n|| rho_h - rho ||_{L^2} = " 
                << solver->calcL2Error(shockExact, 0) << '\n' << endl;
      *out << "\nfinal residual norm = " << solver->calcResidualNorm() << endl;

   }
   catch (MISOException &exception)
   {
      exception.print_message();
   }
   catch (std::exception &exception)
   {
      cerr << exception.what() << endl;
   }

   MPI_Finalize();
}

Mesh buildCurvilinearMesh(int degree, int num_x, int num_y)
{
   Mesh mesh = Mesh::MakeCartesian2D(num_x, num_y, Element::TRIANGLE,
                                     true /* gen. edges */, 1.0, 1.0, true);
   // strategy:
   // 1) generate a fes for Lagrange elements of desired degree
   // 2) create a Grid Function using a VectorFunctionCoefficient
   // 4) use mesh_ptr->NewNodes(nodes, true) to set the mesh nodes
   
   // Problem: fes does not own fec, which is generated in this function's scope
   // Solution: the grid function can own both the fec and fes
   H1_FECollection *fec = new H1_FECollection(degree, 2 /* = dim */);
   FiniteElementSpace *fes = new FiniteElementSpace(&mesh, fec, 2,
                                                    Ordering::byVDIM);

   auto xy_fun = [](const Vector& xi, Vector &x)
   {
      double len_x = 3.0;
      double len_y = 0.3;
      x(0) = xi(0) + (1.0/40.0)*sin(2.0*M_PI*xi(0))*sin(2.0*M_PI*xi(1));
      x(1) = xi(1) + (1.0/40.0)*sin(2.0*M_PI*xi(1))*sin(2.0*M_PI*xi(0));
      x(0) = len_x*x(0) - 0.5*len_x;
      x(1) = len_y*x(1) - 0.5*len_y;
   };
   VectorFunctionCoefficient xy_coeff(2, xy_fun);
   GridFunction *xy = new GridFunction(fes);
   xy->MakeOwner(fec);
   xy->ProjectCoefficient(xy_coeff);

   mesh.NewNodes(*xy, true);
   return mesh;
}