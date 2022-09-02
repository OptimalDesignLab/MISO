/// Solve the steady isentropic vortex problem on a quarter annulus

// set this const expression to true in order to use entropy variables for state
constexpr bool entvar = false;

#include <random>
#include "adept.h"

#include "mfem.hpp"
#include "euler_dg.hpp"
#include <fstream>
#include <iostream>

using namespace std;
using namespace mfem;
using namespace mach;

std::default_random_engine gen(std::random_device{}());
std::uniform_real_distribution<double> normal_rand(-1.0, 1.0);

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
Mesh buildQuarterAnnulusMesh(int degree, int num_rad, int num_ang);

int main(int argc, char *argv[])
{
   const char *options_file = "airfoil_steady_dg_mms_options.json";
   // Initialize MPI
   int num_procs, rank;
   MPI_Init(&argc, &argv);
   MPI_Comm_size(MPI_COMM_WORLD, &num_procs);
   MPI_Comm_rank(MPI_COMM_WORLD, &rank);
   ostream *out = getOutStream(rank);

   // Parse command-line options
   OptionsParser args(argc, argv);
   args.AddOption(&options_file, "-o", "--options", "Options file to use.");
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
      // construct the solver and set initial conditions
      auto solver = createSolver<EulerDGSolver<2, entvar>>(opt_file_name);
      solver->setInitialCondition(uexact);
      solver->printSolution("airfoil_dg_mms_init", 0);

      // get the initial density error
      double l2_error = (static_cast<EulerDGSolver<2, entvar> &>(*solver)
                             .calcConservativeVarsL2Error(uexact, 0));
      double res_error = solver->calcResidualNorm();
      *out << "\n|| rho_h - rho ||_{L^2} = " << l2_error;
      *out << "\ninitial residual norm = " << res_error << endl;
      //solver->printResidual("residual-dg-mms-init", 0);
      // solver->checkJacobian(pert);
      solver->solveForState();
      solver->printSolution("airfoil_dg_mms_final", 0);
      // get the final density error
      l2_error = (static_cast<EulerDGSolver<2, entvar> &>(*solver)
                      .calcConservativeVarsL2Error(uexact, 0));
      res_error = solver->calcResidualNorm();
      auto drag_opts = R"({ "boundaries": [0, 0, 1, 1]})"_json;
      solver->createOutput("drag", drag_opts);
      double drag = abs(solver->calcOutput("drag"));
      // double entropy = solver->calcOutput("entropy");
      out->precision(15);
      *out << "\nfinal residual norm = " << res_error;
      *out << "\n|| rho_h - rho ||_{L^2} = " << l2_error << endl;
      *out << "\nDrag error = " << drag << endl;
      // *out << "\nTotal entropy = " << entropy;
      // *out << "\nEntropy error = "
      //      << fabs(entropy - calcEntropyTotalExact()) << endl;
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
void pert(const Vector &x, Vector &p)
{
   p.SetSize(4);
   for (int i = 0; i < 4; i++)
   {
      p(i) = normal_rand(gen);
   }
}

// Returns the exact total entropy value over the quarter annulus
// Note: the number 8.74655... that appears below is the integral of r*rho over
// the radii from 1 to 3.  It was approixmated using a degree 51 Gaussian
// quadrature.
double calcEntropyTotalExact()
{
   double rhoi = 2.0;
   double prsi = 1.0 / euler::gamma;
   double si = log(prsi / pow(rhoi, euler::gamma));
   return -si * 8.746553803443305 * M_PI * 0.5 / 0.4;
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
   const double scale = 60.0;
   const double trans = 30.0;
   /// define the exact solution
   double rho = rho0 + rhop * pow(sin(M_PI * (x(0) + trans) / scale), 2) *
                           sin(M_PI * (x(1) + trans) / scale);
   double ux =
       4.0 * u0 * ((x(1) + trans) / scale) * (1.0 - (x(1) + trans)/ scale) +
       (up * sin(2.0 * M_PI * ((x(1) + trans)) / scale) * pow(sin(M_PI * (x(0)+ trans) / scale), 2));
   double uy =
       -up * pow(sin(2.0 * M_PI * (x(0) + trans) / scale), 2) * sin(M_PI * (x(1) + trans) / scale);
   double T = T0 + Tp * (pow((x(0) + trans) / scale, 4) - (2.0 * pow((x(0) + trans) / scale, 3)) +
                         pow((x(0) + trans) / scale, 2) + pow((x(1) + trans) / scale, 4) -
                         (2.0 * pow((x(1) + trans) / scale, 3)) + pow((x(1) + trans) / scale, 2));
   double p = rho * T;
   double e = (p / (euler::gamma - 1)) + 0.5 * rho * (ux * ux + uy * uy);
   u(0) = rho;
   u(1) = ux;  // multiply by rho ?
   u(2) = uy;
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
