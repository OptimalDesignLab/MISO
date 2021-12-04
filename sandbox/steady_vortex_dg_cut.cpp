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
void uexact(const Vector &x, Vector &u);
/// Generate quarter annulus mesh
/// \param[in] N - number of elements in x-y direction
Mesh buildMesh(int N);
int main(int argc, char *argv[])
{
   const char *options_file = "steady_vortex_dg_cut_options.json";
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
      ofstream sol_ofs("cart_mesh_dg_cut.vtk");
      sol_ofs.precision(14);
      smesh->PrintVTK(sol_ofs, 0);
      string opt_file_name(options_file);
      auto solver =
          createSolver<CutEulerDGSolver<2, entvar>>(opt_file_name, move(smesh));
      solver->setInitialCondition(uexact);
      solver->printSolution("vortex-steady-dg-cut-init");
      // get the initial density error
      double l2_error = (static_cast<CutEulerDGSolver<2, entvar> &>(*solver)
                             .calcConservativeVarsL2Error(uexact, 0));
      double res_error = solver->calcResidualNorm();
      *out << "\n|| rho_h - rho ||_{L^2} = " << l2_error;
      *out << "\ninitial residual norm = " << res_error << endl;
      solver->solveForState();
      solver->printSolution("vortex-dg-cut-final", 0);
      // get the final density error
      l2_error = (static_cast<CutEulerDGSolver<2, entvar> &>(*solver)
                      .calcConservativeVarsL2Error(uexact, 0));
      res_error = solver->calcResidualNorm();
      out->precision(15);
      *out << "\nfinal residual norm = " << res_error << endl;
      auto drag_opts = R"({ "boundaries": [1, 0, 0, 1]})"_json;
      solver->createOutput("drag", drag_opts);
      double drag = abs(solver->calcOutput("drag") - (-1 / mach::euler::gamma));
      cout << "================================================================"
           << endl;
      *out << "    || rho_h - rho ||_{L^2} = " << l2_error << endl;
      *out << "    Drag error = " << drag << endl;
      cout << "================================================================"
           << endl;
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
void uexact(const Vector &x, Vector &q)
{
   q.SetSize(4);
   Vector u(4);
   double ri = 1.0;
   double Mai = 0.5;  // 0.95
   double rhoi = 2.0;
   double prsi = 1.0 / euler::gamma;
   double rinv = ri / sqrt(x(0) * x(0) + x(1) * x(1));
   double rho =
       rhoi * pow(1.0 + 0.5 * euler::gami * Mai * Mai * (1.0 - rinv * rinv),
                  1.0 / euler::gami);
   double Ma =
       sqrt((2.0 / euler::gami) * ((pow(rhoi / rho, euler::gami)) *
                                       (1.0 + 0.5 * euler::gami * Mai * Mai) -
                                   1.0));
   double theta;
   if (x(0) > 1e-15)
   {
      theta = atan(x(1) / x(0));
   }
   else
   {
      theta = M_PI / 2.0;
   }
   double press = prsi * pow((1.0 + 0.5 * euler::gami * Mai * Mai) /
                                 (1.0 + 0.5 * euler::gami * Ma * Ma),
                             euler::gamma / euler::gami);
   double a = sqrt(euler::gamma * press / rho);

   u(0) = rho;
   u(1) = -rho * a * Ma * sin(theta);
   u(2) = rho * a * Ma * cos(theta);
   u(3) = press / euler::gami + 0.5 * rho * a * a * Ma * Ma;

   if (entvar == false)
   {
      q = u;
   }
   else
   {
      calcEntropyVars<double, 2>(u.GetData(), q.GetData());
   }
}
Mesh buildMesh(int N)
{
   Mesh mesh = Mesh::MakeCartesian2D(
       N, N, Element::QUADRILATERAL, true, 3.0, 3.0, true);
   return mesh;
}
