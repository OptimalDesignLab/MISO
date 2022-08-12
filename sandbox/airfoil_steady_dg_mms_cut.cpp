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

/// Generate quarter annulus mesh
/// \param[in] N - number of elements in x-y direction
Mesh buildMesh(int N);
int main(int argc, char *argv[])
{
   const char *options_file = "airfoil_steady_dg_cut_options.json";
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
      solver->setInitialCondition(uexact);
      // Vector wfar(4);
      // TODO: I do not like that we have to perform this conversion outside the
      // solver...
      // calcEntropyVars<double, 2>(qfar.GetData(), wfar.GetData());
      //solver->setInitialCondition(qfar);
      solver->printSolution("airfoil-steady-dg-cut-mms-init", 0);
      // solver->checkJacobian(pert);
      // solver->printResidual("residual-init", 0);
      // solver->calcResidualNorm();
      mfem::out << "\ninitial residual norm = " << solver->calcResidualNorm()
                << endl;
      solver->solveForState();
      solver->printSolution("airfoil-steady-dg-cut-mms-final", 0);
      mfem::out << "\nfinal residual norm = " << solver->calcResidualNorm()
                << endl;
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
   /// define the exact solution
   double rho = rho0 + rhop * pow(sin(M_PI * x(0)), 2) * sin(M_PI * x(1));
   double ux = 4.0 * u0 * x(1) * (1.0 - x(1)) +
               (up * sin(2.0 * M_PI * x(1)) * pow(sin(M_PI * x(0)), 2));
   double uy = -up * pow(sin(2.0 * M_PI * x(0)), 2) * sin(M_PI * x(1));
   double T = T0 + Tp * (pow(x(0), 4) - (2.0 * pow(x(0), 3)) + pow(x(0), 2) +
                         pow(x(1), 4) - (2.0 * pow(x(1), 3)) + pow(x(1), 2));
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

/// source term
void fsource(const Vector &x, Vector &src)
{
   using euler::gamma;
   const double rho0 = 1.0;
   const double rhop = 0.05;
   const double u0 = 0.5;
   const double up = 0.05;
   const double T0 = 1.0;
   const double Tp = 0.05;
   src.SetSize(4);
   src(0) =
       M_PI *
       (-up * rhop * pow(sin(M_PI * x[0]), 2) * pow(sin(2 * M_PI * x[0]), 2) *
            sin(M_PI * x[1]) * cos(M_PI * x[1]) +
        2 * up * (rho0 + rhop * pow(sin(M_PI * x[0]), 2) * sin(M_PI * x[1])) *
            sin(M_PI * x[0]) * sin(2 * M_PI * x[1]) * cos(M_PI * x[0]) -
        up * (rho0 + rhop * pow(sin(M_PI * x[0]), 2) * sin(M_PI * x[1])) *
            pow(sin(2 * M_PI * x[0]), 2) * cos(M_PI * x[1]) -
        2 * rhop *
            (4 * u0 * x[1] * (x[1] - 1) -
             up * pow(sin(M_PI * x[0]), 2) * sin(2 * M_PI * x[1])) *
            sin(M_PI * x[0]) * sin(M_PI * x[1]) * cos(M_PI * x[0]));
   src(1) =
       2 * Tp * x[0] *
           (rho0 + rhop * pow(sin(M_PI * x[0]), 2) * sin(M_PI * x[1])) *
           (2 * pow(x[0], 2) - 3 * x[0] + 1) +
       M_PI * up * rhop *
           (4 * u0 * x[1] * (x[1] - 1) -
            up * pow(sin(M_PI * x[0]), 2) * sin(2 * M_PI * x[1])) *
           pow(sin(M_PI * x[0]), 2) * pow(sin(2 * M_PI * x[0]), 2) *
           sin(M_PI * x[1]) * cos(M_PI * x[1]) -
       4 * M_PI * up *
           (rho0 + rhop * pow(sin(M_PI * x[0]), 2) * sin(M_PI * x[1])) *
           (4 * u0 * x[1] * (x[1] - 1) -
            up * pow(sin(M_PI * x[0]), 2) * sin(2 * M_PI * x[1])) *
           sin(M_PI * x[0]) * sin(2 * M_PI * x[1]) * cos(M_PI * x[0]) +
       M_PI * up * (rho0 + rhop * pow(sin(M_PI * x[0]), 2) * sin(M_PI * x[1])) *
           (4 * u0 * x[1] * (x[1] - 1) -
            up * pow(sin(M_PI * x[0]), 2) * sin(2 * M_PI * x[1])) *
           pow(sin(2 * M_PI * x[0]), 2) * cos(M_PI * x[1]) +
       2 * up * (rho0 + rhop * pow(sin(M_PI * x[0]), 2) * sin(M_PI * x[1])) *
           (2 * u0 * x[1] + 2 * u0 * (x[1] - 1) -
            M_PI * up * pow(sin(M_PI * x[0]), 2) * cos(2 * M_PI * x[1])) *
           pow(sin(2 * M_PI * x[0]), 2) * sin(M_PI * x[1]) +
       (1.0 / 2.0) * M_PI * rhop *
           (T0 + Tp * (pow(x[0], 4) - 2 * pow(x[0], 3) + pow(x[0], 2) +
                       pow(x[1], 4) - 2 * pow(x[1], 3) + pow(x[1], 2))) *
           (cos(M_PI * (2 * x[0] - x[1])) - cos(M_PI * (2 * x[0] + x[1]))) +
       2 * M_PI * rhop *
           pow(4 * u0 * x[1] * (x[1] - 1) -
                   up * pow(sin(M_PI * x[0]), 2) * sin(2 * M_PI * x[1]),
               2) *
           sin(M_PI * x[0]) * sin(M_PI * x[1]) * cos(M_PI * x[0]);
   src(2) = 2 * Tp * x[1] *
                (rho0 + rhop * pow(sin(M_PI * x[0]), 2) * sin(M_PI * x[1])) *
                (2 * pow(x[1], 2) - 3 * x[1] + 1) +
            M_PI * pow(up, 2) * rhop * pow(sin(M_PI * x[0]), 2) *
                pow(sin(2 * M_PI * x[0]), 4) * pow(sin(M_PI * x[1]), 2) *
                cos(M_PI * x[1]) -
            16 * M_PI * pow(up, 2) *
                (rho0 + rhop * pow(sin(M_PI * x[0]), 2) * sin(M_PI * x[1])) *
                pow(sin(M_PI * x[0]), 3) * pow(sin(M_PI * x[1]), 2) *
                pow(cos(M_PI * x[0]), 3) * cos(M_PI * x[1]) +
            2 * M_PI * pow(up, 2) *
                (rho0 + rhop * pow(sin(M_PI * x[0]), 2) * sin(M_PI * x[1])) *
                pow(sin(2 * M_PI * x[0]), 4) * sin(M_PI * x[1]) *
                cos(M_PI * x[1]) +
            2 * M_PI * up * rhop *
                (4 * u0 * x[1] * (x[1] - 1) -
                 up * pow(sin(M_PI * x[0]), 2) * sin(2 * M_PI * x[1])) *
                sin(M_PI * x[0]) * pow(sin(2 * M_PI * x[0]), 2) *
                pow(sin(M_PI * x[1]), 2) * cos(M_PI * x[0]) +
            4 * M_PI * up *
                (rho0 + rhop * pow(sin(M_PI * x[0]), 2) * sin(M_PI * x[1])) *
                (4 * u0 * x[1] * (x[1] - 1) -
                 up * pow(sin(M_PI * x[0]), 2) * sin(2 * M_PI * x[1])) *
                sin(2 * M_PI * x[0]) * sin(M_PI * x[1]) * cos(2 * M_PI * x[0]) +
            M_PI * rhop *
                (T0 + Tp * (pow(x[0], 4) - 2 * pow(x[0], 3) + pow(x[0], 2) +
                            pow(x[1], 4) - 2 * pow(x[1], 3) + pow(x[1], 2))) *
                pow(sin(M_PI * x[0]), 2) * cos(M_PI * x[1]);
   src(4) =
       (M_PI * up *
            (rho0 + rhop * pow(sin(M_PI * x[0]), 2) * sin(M_PI * x[1])) *
            (2 * T0 +
             2 * Tp *
                 (pow(x[0], 4) - 2 * pow(x[0], 3) + pow(x[0], 2) +
                  pow(x[1], 4) - 2 * pow(x[1], 3) + pow(x[1], 2)) +
             (gamma - 1) *
                 (2 * T0 +
                  2 * Tp *
                      (pow(x[0], 4) - 2 * pow(x[0], 3) + pow(x[0], 2) +
                       pow(x[1], 4) - 2 * pow(x[1], 3) + pow(x[1], 2)) +
                  pow(up, 2) * pow(sin(2 * M_PI * x[0]), 4) *
                      pow(sin(M_PI * x[1]), 2) +
                  pow(4 * u0 * x[1] * (x[1] - 1) -
                          up * pow(sin(M_PI * x[0]), 2) * sin(2 * M_PI * x[1]),
                      2))) *
            sin(M_PI * x[0]) * sin(2 * M_PI * x[1]) * cos(M_PI * x[0]) -
        1.0 / 2.0 * M_PI * up *
            (rho0 + rhop * pow(sin(M_PI * x[0]), 2) * sin(M_PI * x[1])) *
            (2 * T0 +
             2 * Tp *
                 (pow(x[0], 4) - 2 * pow(x[0], 3) + pow(x[0], 2) +
                  pow(x[1], 4) - 2 * pow(x[1], 3) + pow(x[1], 2)) +
             (gamma - 1) *
                 (2 * T0 +
                  2 * Tp *
                      (pow(x[0], 4) - 2 * pow(x[0], 3) + pow(x[0], 2) +
                       pow(x[1], 4) - 2 * pow(x[1], 3) + pow(x[1], 2)) +
                  pow(up, 2) * pow(sin(2 * M_PI * x[0]), 4) *
                      pow(sin(M_PI * x[1]), 2) +
                  pow(4 * u0 * x[1] * (x[1] - 1) -
                          up * pow(sin(M_PI * x[0]), 2) * sin(2 * M_PI * x[1]),
                      2))) *
            pow(sin(2 * M_PI * x[0]), 2) * cos(M_PI * x[1]) -
        1.0 / 2.0 * up *
            (4 * Tp * x[1] *
                 (rho0 + rhop * pow(sin(M_PI * x[0]), 2) * sin(M_PI * x[1])) *
                 (2 * pow(x[1], 2) - 3 * x[1] + 1) +
             2 * M_PI * rhop *
                 (T0 + Tp * (pow(x[0], 4) - 2 * pow(x[0], 3) + pow(x[0], 2) +
                             pow(x[1], 4) - 2 * pow(x[1], 3) + pow(x[1], 2))) *
                 pow(sin(M_PI * x[0]), 2) * cos(M_PI * x[1]) +
             (gamma - 1) *
                 (4 * Tp * x[1] *
                      (rho0 +
                       rhop * pow(sin(M_PI * x[0]), 2) * sin(M_PI * x[1])) *
                      (2 * pow(x[1], 2) - 3 * x[1] + 1) +
                  2 * M_PI * rhop *
                      (T0 +
                       Tp * (pow(x[0], 4) - 2 * pow(x[0], 3) + pow(x[0], 2) +
                             pow(x[1], 4) - 2 * pow(x[1], 3) + pow(x[1], 2))) *
                      pow(sin(M_PI * x[0]), 2) * cos(M_PI * x[1]) +
                  M_PI * rhop *
                      (pow(up, 2) * pow(sin(2 * M_PI * x[0]), 4) *
                           pow(sin(M_PI * x[1]), 2) +
                       pow(4 * u0 * x[1] * (x[1] - 1) -
                               up * pow(sin(M_PI * x[0]), 2) *
                                   sin(2 * M_PI * x[1]),
                           2)) *
                      pow(sin(M_PI * x[0]), 2) * cos(M_PI * x[1]) +
                  2 *
                      (rho0 +
                       rhop * pow(sin(M_PI * x[0]), 2) * sin(M_PI * x[1])) *
                      (M_PI * pow(up, 2) * pow(sin(2 * M_PI * x[0]), 4) *
                           sin(M_PI * x[1]) * cos(M_PI * x[1]) +
                       2 *
                           (4 * u0 * x[1] * (x[1] - 1) -
                            up * pow(sin(M_PI * x[0]), 2) *
                                sin(2 * M_PI * x[1])) *
                           (2 * u0 * x[1] + 2 * u0 * (x[1] - 1) -
                            M_PI * up * pow(sin(M_PI * x[0]), 2) *
                                cos(2 * M_PI * x[1]))))) *
            pow(sin(2 * M_PI * x[0]), 2) * sin(M_PI * x[1]) -
        (4 * u0 * x[1] * (x[1] - 1) -
         up * pow(sin(M_PI * x[0]), 2) * sin(2 * M_PI * x[1])) *
            (2 * Tp * x[0] *
                 (rho0 + rhop * pow(sin(M_PI * x[0]), 2) * sin(M_PI * x[1])) *
                 (2 * pow(x[0], 2) - 3 * x[0] + 1) +
             (1.0 / 2.0) * M_PI * rhop *
                 (T0 + Tp * (pow(x[0], 4) - 2 * pow(x[0], 3) + pow(x[0], 2) +
                             pow(x[1], 4) - 2 * pow(x[1], 3) + pow(x[1], 2))) *
                 (cos(M_PI * (2 * x[0] - x[1])) -
                  cos(M_PI * (2 * x[0] + x[1]))) +
             (gamma - 1) *
                 (2 * Tp * x[0] *
                      (rho0 +
                       rhop * pow(sin(M_PI * x[0]), 2) * sin(M_PI * x[1])) *
                      (2 * pow(x[0], 2) - 3 * x[0] + 1) +
                  2 * M_PI * up *
                      (rho0 +
                       rhop * pow(sin(M_PI * x[0]), 2) * sin(M_PI * x[1])) *
                      (2 * up * pow(sin(2 * M_PI * x[0]), 3) *
                           pow(sin(M_PI * x[1]), 2) * cos(2 * M_PI * x[0]) -
                       (4 * u0 * x[1] * (x[1] - 1) -
                        up * pow(sin(M_PI * x[0]), 2) * sin(2 * M_PI * x[1])) *
                           sin(M_PI * x[0]) * sin(2 * M_PI * x[1]) *
                           cos(M_PI * x[0])) +
                  (1.0 / 2.0) * M_PI * rhop *
                      (T0 +
                       Tp * (pow(x[0], 4) - 2 * pow(x[0], 3) + pow(x[0], 2) +
                             pow(x[1], 4) - 2 * pow(x[1], 3) + pow(x[1], 2))) *
                      (cos(M_PI * (2 * x[0] - x[1])) -
                       cos(M_PI * (2 * x[0] + x[1]))) +
                  M_PI * rhop *
                      (pow(up, 2) * pow(sin(2 * M_PI * x[0]), 4) *
                           pow(sin(M_PI * x[1]), 2) +
                       pow(4 * u0 * x[1] * (x[1] - 1) -
                               up * pow(sin(M_PI * x[0]), 2) *
                                   sin(2 * M_PI * x[1]),
                           2)) *
                      sin(M_PI * x[0]) * sin(M_PI * x[1]) *
                      cos(M_PI * x[0])))) /
       (gamma - 1);
}
