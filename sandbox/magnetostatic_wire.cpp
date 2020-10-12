#include "mfem.hpp"
#include "magnetostatic.hpp"

#include <fstream>
#include <iostream>

using namespace std;
using namespace mfem;
using namespace mach;

int main(int argc, char *argv[])
{
   // Initialize MPI
   MPI_Init(&argc, &argv);
   int rank;
   MPI_Comm_rank(MPI_COMM_WORLD, &rank);
   ostream *out = getOutStream(rank);

   // Parse command-line options
   OptionsParser args(argc, argv);
   const char *options_file = "magnetostatic_wire_options.json";
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
      // construct the solver
      string opt_file_name(options_file);

      auto solver = createSolver<MagnetostaticSolver>(opt_file_name);
      solver->solveForState();
      std::cout << "finish steady solve\n";
      
      double coenergy = solver->calcOutput("co-energy");
      std::cout << "Co-energy = " << coenergy << std::endl;
      solver->solveForAdjoint("co-energy");
      solver->printAdjoint("co-energy-adjoint");
      solver->printSolution("wire_out");
      solver->verifyMeshSensitivities();
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

