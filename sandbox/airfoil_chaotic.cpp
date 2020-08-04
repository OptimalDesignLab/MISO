// Solve for the flow around a NACA0012 at high angle of attack

#include <fstream>
#include <iostream>

#include "mfem.hpp"

#include "mach.hpp"

using namespace std;
using namespace mfem;
using namespace mach;

void uinit(const Vector &x, Vector& u0);

int main(int argc, char *argv[])
{
   const char *options_file = "airfoil_chaotic_options.json";
   // Initialize MPI
   MPI_Init(&argc, &argv);

   // Parse command-line options
   OptionsParser args(argc, argv);
   args.AddOption(&options_file, "-o", "--options",
                  "Options file to use.");
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
#if 1
      auto solver = createSolver<NavierStokesSolver<2>>(opt_file_name);
#endif
#if 0
      auto solver = createSolver<EulerSolver<2>>(opt_file_name);
#endif
      Vector qfar(4);
      dynamic_cast<EulerSolver<2>*>(solver.get())->getFreeStreamState(qfar);

      // TEMP?
      qfar(1) = 0.0;
      qfar(2) = 0.0;

      //static_cast<EulerSolver<2>*>(solver)->getFreeStreamState(qfar); 
      solver->setInitialCondition(qfar);
      solver->printSolution("airfoil-chaotic-init");

      mfem::out << "\ninitial residual norm = " << solver->calcResidualNorm()
                << endl;

      solver->solveForState();
      solver->printSolution("airfoil-chaotic-final");

      mfem::out << "\nfinal residual norm = " << solver->calcResidualNorm()
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

void uinit(const Vector &x, Vector& u0)
{
   u0.SetSize(4);
   u0 = 1.0;
}
