// Solve for the steady flow around a NACA0012

// set this const expression to true in order to use entropy variables for state
constexpr bool entvar = false;

#include <random>
#include <fstream>
#include <iostream>

#include "mfem.hpp"

#include "euler_dg.hpp"

using namespace std;
using namespace mfem;
using namespace mach;

std::default_random_engine gen(std::random_device{}());
std::uniform_real_distribution<double> normal_rand(-1.0, 1.0);

/// \brief Defines the random function for the jabocian check
/// \param[in] x - coordinate of the point at which the state is needed
/// \param[out] u - conservative variables stored as a 4-vector
void pert(const Vector &x, Vector &p);

int main(int argc, char *argv[])
{
   const char *options_file = "airfoil_steady_dg_options.json";
   // Initialize MPI
   MPI_Init(&argc, &argv);

   // Parse command-line options
   OptionsParser args(argc, argv);
   args.AddOption(&options_file, "-o", "--options", "Options file to use.");
   args.Parse();
   if (!args.Good())
   {
      args.PrintUsage(cout);
      return 1;
   }

   try
   {
      // construct the solver, set the initial condition, and solve
      string opt_file_name(options_file);
      auto solver = createSolver<EulerDGSolver<2, entvar>>(opt_file_name);
      Vector qfar(4);
      static_cast<EulerDGSolver<2, entvar> *>(solver.get())
          ->getFreeStreamState(qfar);
      qfar.Print();
      // Vector wfar(4);
      // TODO: I do not like that we have to perform this conversion outside the
      // solver...
      // calcEntropyVars<double, 2>(qfar.GetData(), wfar.GetData());
      solver->setInitialCondition(qfar);
      solver->printSolution("airfoil-steady-dg-init");
      //solver->checkJacobian(pert);
     // solver->printResidual("residual-init");
      mfem::out << "\ninitial residual norm = " << solver->calcResidualNorm()
                << endl;
      // solver->solveForState();
      // solver->printSolution("airfoil-steady-dg-final");
      // mfem::out << "\nfinal residual norm = " << solver->calcResidualNorm()
      //           << endl;
      // auto drag_opts = R"({ "boundaries": [0, 0, 1, 1]})"_json;
      // solver->createOutput("drag", drag_opts);
      // double drag = abs(solver->calcOutput("drag"));
      // mfem::out << "\nDrag error = " << drag << endl;
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