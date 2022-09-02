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

/// \brief Defines the random function for the jabocian check
/// \param[in] x - coordinate of the point at which the state is needed
/// \param[out] u - conservative variables stored as a 4-vector
void pert(const Vector &x, Vector &p);
/// \brief Defines the exact solution for the steady isentropic vortex
/// \param[in] x - coordinate of the point at which the state is needed
/// \param[out] u - state variables stored as a 4-vector
void uexact(const Vector &x, Vector &q);
/// Generate quarter annulus mesh
/// \param[in] N - number of elements in x-y direction
Mesh buildMesh(int N);
int main(int argc, char *argv[])
{
   const char *options_file = "airfoil_steady_dg_mms_cut_options.json";
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
      //   Vector qfar(4);
      //   static_cast<CutEulerDGSolver<2, entvar> *>(solver.get())
      //       ->getFreeStreamState(qfar);
      //   qfar.Print();
      //       solver->setInitialCondition(qfar);
      solver->setInitialCondition(uexact);
      solver->printSolution("airfoil-steady-dg-cut-mms-init", 0);
      //solver->checkJacobian(pert);
      // solver->printResidual("residual-init", 0);
      mfem::out << "\ninitial residual norm = " << solver->calcResidualNorm()
                << endl;
      // get the initial density error
      double l2_error = (static_cast<CutEulerDGSolver<2, entvar> &>(*solver)
                             .calcConservativeVarsL2Error(uexact, 0));

      *out << "\n|| rho_h - rho ||_{L^2} = " << l2_error << endl;
      solver->solveForState();
      solver->printSolution("airfoil-steady-dg-cut-mms-final", 0);
      mfem::out << "\nfinal residual norm = " << solver->calcResidualNorm()
                << endl;
      l2_error = (static_cast<CutEulerDGSolver<2, entvar> &>(*solver)
                      .calcConservativeVarsL2Error(uexact, 0));
      *out << "\n|| rho_h - rho ||_{L^2} = " << l2_error << endl;
      auto drag_opts = R"({ "boundaries": [0, 0, 1, 1]})"_json;
      solver->createOutput("drag", drag_opts);
      double drag = abs(solver->calcOutput("drag"));
      mfem::out << "\nDrag error = " << drag << endl;
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
       N, N, Element::QUADRILATERAL, true, 20.0, 20.0, true);
   return mesh;
}

/// Exact solution
void uexact(const Vector &x, Vector &q)
{
   q.SetSize(4);
   Vector u(4);
   const double rho0 = 1.0;
   const double rhop = 0.05;
   const double u0 = 0.5;
   const double up = 0.05;
   const double T0 = 1.0;
   const double Tp = 0.05;
   const double scale = 20.0;
   /// define the exact solution
   double rho = rho0 + rhop * pow(sin(M_PI * x(0) / scale), 2) *
                           sin(M_PI * x(1) / scale);
   double ux =
       4.0 * u0 * (x(1) / scale) * (1.0 - x(1) / scale) +
       (up * sin(2.0 * M_PI * x(1) / scale) * pow(sin(M_PI * x(0) / scale), 2));
   double uy =
       -up * pow(sin(2.0 * M_PI * x(0) / scale), 2) * sin(M_PI * x(1) / scale);
   double T = T0 + Tp * (pow(x(0) / scale, 4) - (2.0 * pow(x(0) / scale, 3)) +
                         pow(x(0) / scale, 2) + pow(x(1) / scale, 4) -
                         (2.0 * pow(x(1) / scale, 3)) + pow(x(1) / scale, 2));
   double p = rho * T;
   double e = (p / (euler::gamma - 1)) + 0.5 * rho * (ux * ux + uy * uy);
   u(0) = rho;
   u(1) = rho*ux;  // multiply by rho ?
   u(2) = rho*uy;
   u(3) = e;
   if (entvar == false)
   {
      q = u;
   }
   else
   {
      calcEntropyVars<double, 2>(u.GetData(), q.GetData());
   }
}