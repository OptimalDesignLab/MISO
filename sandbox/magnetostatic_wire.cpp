#include "mfem.hpp"
#include "magnetostatic.hpp"

#include <fstream>
#include <iostream>

using namespace std;
using namespace mfem;
using namespace mach;

int main(int argc, char *argv[])
{
   ostream *out;
#ifdef MFEM_USE_MPI
   // Initialize MPI if parallel
   MPI_Init(&argc, &argv);
   int rank;
   MPI_Comm_rank(MPI_COMM_WORLD, &rank);
   out = getOutStream(rank); 
#else
   out = getOutStream(0);
#endif

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
      MagnetostaticSolver solver(opt_file_name);
      solver.initDerived();
      solver.solveForState();
      std::cout << "finish steady solve\n";
      double energy = solver.calcOutput("energy");
      std::cout << "Energy = " << energy << std::endl;
      double coenergy = solver.calcOutput("co-energy");
      std::cout << "Co-energy = " << coenergy << std::endl;
      solver.solveForAdjoint("co-energy");
      solver.printAdjoint("co-energy-adjoint");
      solver.printSolution("wire_out");
   }
   catch (MachException &exception)
   {
      exception.print_message();
   }
   catch (std::exception &exception)
   {
      cerr << exception.what() << endl;
   }
#ifdef MFEM_USE_MPI
   MPI_Finalize();
#endif
}

