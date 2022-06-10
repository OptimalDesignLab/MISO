// Solve for the steady flow around a NACA0012

#include <fstream>
#include <iostream>

#include "mach.hpp"

using namespace std;
using namespace mfem;
using namespace mach;

int main(int argc, char *argv[])
{
   // Get the options
   const char *options_file = "airfoil_steady_options.json";
   nlohmann::json options;
   ifstream option_source(options_file);
   option_source >> options;

   // Initialize MPI
   int num_procs, rank;
   MPI_Init(&argc, &argv);
   MPI_Comm_size(MPI_COMM_WORLD, &num_procs);
   MPI_Comm_rank(MPI_COMM_WORLD, &rank);
   ostream *out = getOutStream(rank);

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

      // Create solver and set initial condition
      FlowSolver<2> solver(MPI_COMM_WORLD, options);
      mfem::Vector state_tv(solver.getStateSize());
      Vector qfar(4);
      solver.getFreeStreamState(qfar);
      auto uInit = [&](const Vector &x, Vector &u0) { u0 = qfar; };
      solver.setState(uInit, state_tv);

      // create the outputs
      solver.createOutput("entropy", options["outputs"].at("entropy"));
      solver.createOutput("drag", options["outputs"].at("drag"));
      solver.createOutput("lift", options["outputs"].at("lift"));

      // Solve for the state
      MachInputs inputs({{"state", state_tv}});
      solver.solveForState(inputs, state_tv);

      // Compute the entropy, lift and drag
      double entropy = solver.calcOutput("entropy", inputs);
      *out << "entropy = " << entropy << endl;
      double lift = solver.calcOutput("lift", inputs);
      *out << "lift = " << lift << endl;
      double drag = solver.calcOutput("drag", inputs);
      *out << "drag = " << drag << endl;
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