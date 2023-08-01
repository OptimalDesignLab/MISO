#include "mfem.hpp"
#include "magnetostatic.hpp"

#include <fstream>
#include <iostream>

using namespace std;
using namespace mfem;
using namespace miso;


int main(int argc, char *argv[])
{
   // Initialize MPI
   MPI_Init(&argc, &argv);
   int rank;
   MPI_Comm_rank(MPI_COMM_WORLD, &rank);
   ostream *out = getOutStream(rank);

   // Parse command-line options
   OptionsParser args(argc, argv);
   const char *options_file = "magnetostatic_motor_options.json";
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
   }
   catch (MISOException &exception)
   {
      exception.print_message();
   }
   catch (std::exception &exception)
   {
      cerr << exception.what() << endl;
   }
   
   MPI_Finalize();
}

