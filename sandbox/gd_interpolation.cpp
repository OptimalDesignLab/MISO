// Solve for the steady flow around a NACA0012

// set this const expression to true in order to use entropy variables for state
constexpr bool entvar = false;

#include <random>
#include <fstream>
#include <iostream>

#include "mfem.hpp"

#include "euler_dg_cut.hpp"

using namespace std;
using namespace mfem;
using namespace mach;

std::default_random_engine gen(std::random_device{}());
std::uniform_real_distribution<double> normal_rand(-1.0, 1.0);

/// \brief Defines the exact solution for the steady isentropic vortex
/// \param[in] x - coordinate of the point at which the state is needed
/// \param[out] u - state variables stored as a 4-vector
void uexact(const Vector &x, Vector &q);

/// Generate quarter annulus mesh
/// \param[in] N - number of elements in x-y direction
Mesh buildMesh(int N);
int main(int argc, char *argv[])
{
   const char *options_file = "gd_interpolation_options.json";
   // Initialize MPI
   int num_procs, rank;
   MPI_Init(&argc, &argv);
   MPI_Comm_size(MPI_COMM_WORLD, &num_procs);
   MPI_Comm_rank(MPI_COMM_WORLD, &rank);
   ostream *out = getOutStream(rank);
   int N = 5;
   // Parse command-line options
   OptionsParser args(argc, argv);
   args.AddOption(&options_file, "-o", "--options", "Options file to use.");
   args.AddOption(&N, "-n", "--#elements", "number of elements");
   args.Parse();
   if (!args.Good())
   {
      args.PrintUsage(cout);
      return 1;
   }

   try
   {
      // construct the solver, set the initial condition, and solve
      unique_ptr<Mesh> smesh(new Mesh(buildMesh(N)));
      *out << "Number of elements " << smesh->GetNE() << '\n';
      ofstream sol_ofs("cart_mesh_dg_cut_init.vtk");
      sol_ofs.precision(14);
      smesh->PrintVTK(sol_ofs, 0);
      string opt_file_name(options_file);
      auto solver =
          createSolver<CutEulerDGSolver<2, entvar>>(opt_file_name, move(smesh));
      solver->setInitialCondition(uexact);
      solver->printSolution("gd-sol-init-f1", 0);
      out->precision(15);
      // solver->checkJacobian(pert);
      //  solver->printResidual("residual-init", 0);
      // mfem::out << "\ninitial residual norm = " << solver->calcResidualNorm()
      //           << endl;
      // get the initial sol error
      double l2_error = (static_cast<CutEulerDGSolver<2, entvar> &>(*solver)
                             .calcConservativeVarsL2Error(uexact, 0));

      *out << "\n|| u_h - u ||_{L^2} = " << l2_error << endl;
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

// perturbation function used to check the jacobian in each iteration
void pert(const Vector &x, Vector &p)
{
   p.SetSize(4);
   for (int i = 0; i < 4; i++)
   {
      p(i) = normal_rand(gen);
   }
}

Mesh buildMesh(int N)
{
   Mesh mesh = Mesh::MakeCartesian2D(
       N, N, Element::QUADRILATERAL, true, 1.0, 5.0, true);
   return mesh;
}

/// Exact solution
void uexact(const Vector &x, Vector &q)
{
   q.SetSize(1);
   Vector u(1);
   double s = 20.0;
   /// f1
   double a = (x(0) * x(0)/(s*s)) + (x(1) * x(1)/(s*s));
   double b = 2.0 * (x(0) * x(0)/(s*s)) + ((x(1)/s - 0.5) * (x(1)/s - 0.5));
   double f1 = sin(2.0 * M_PI * a) - sin(2.0 * M_PI * b);
   /// f2
   a = (x(0)/s - 0.5);
   b = (x(1)/s - 0.5);
   double f2 = exp(-(a * a + b * b));
   /// f3
   double f3 = sin(2.0 * x(0)/s) + exp(-x(0)/s);
   /// f4
   double f4 = 1.0 + sin(4.0 * x(0)/s) + cos(3.0 * x(0)/s) + sin(2.0 * x(1)/s);
   u(0) = f1;
   q = u;
}