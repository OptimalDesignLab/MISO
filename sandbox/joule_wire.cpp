#include <fstream>
#include <iostream>

#include "mfem.hpp"
#include "mach.hpp"

// using namespace std;
// using namespace mfem;
// using namespace mach;

static double theta0;
static double t_final;
static double initialTemperature(const mfem::Vector &x);

int main(int argc, char *argv[])
{
   std::ostream *out;
#ifdef MFEM_USE_MPI
   // Initialize MPI if parallel
   MPI_Init(&argc, &argv);
   int rank;
   MPI_Comm_rank(MPI_COMM_WORLD, &rank);
   out = mach::getOutStream(rank); 
#else
   out = mach::getOutStream(0);
#endif

   // Parse command-line options
   mfem::OptionsParser args(argc, argv);
   const char *options_file = "joule_wire_options.json";
   args.AddOption(&options_file, "-o", "--options",
                  "Options file to use.");
   args.Parse();
   if (!args.Good())
   {
      args.PrintUsage(*out);
      return 1;
   }

   std::string opt_file_name(options_file);
   nlohmann::json file_options;
   std::ifstream opts(opt_file_name);
   opts >> file_options;

   file_options["problem-opts"]["init-temp"].get_to(theta0);
   file_options["thermal-opts"]["time-dis"]["t-final"].get_to(t_final);

   try
   {
      // construct the solver
      mach::JouleSolver solver(opt_file_name);
      solver.setInitialCondition(initialTemperature);
      *out << "Solving..." << std::endl;
      solver.solveForState();
      *out << "Solving done." << std::endl;
   }
   catch (mach::MachException &exception)
   {
      exception.print_message();
   }
   catch (std::exception &exception)
   {
      std::cerr << exception.what() << std::endl;
   }
#ifdef MFEM_USE_MPI
   MPI_Finalize();
#endif
}

double initialTemperature(const mfem::Vector &x)
{
   // 70 deg Fahrenheit
   return theta0;
}
