#include "mfem.hpp"
#include "thermal.hpp"
#include "gmi_egads.h"
#include "miso_egads.hpp"
#include "mesh_movement.hpp"

#include <fstream>
#include <iostream>

using namespace std;
using namespace mfem;
using namespace miso;

#ifdef MFEM_USE_PUMI
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
   const char *options_file = "mesh_move_2_options.json";
   args.AddOption(&options_file, "-o", "--options",
                  "Options file to use.");
   args.Parse();
   if (!args.Good())
   {
      args.PrintUsage(cout);
      return 1;
   }

   string opt_file_name(options_file);
   nlohmann::json options;
   nlohmann::json file_options;
   ifstream opts(opt_file_name);
   opts >> file_options;
   options.merge_patch(file_options);

   string model_file_old = options["model-file"].template get<string>();
   string model_file_new = options["model-file-new"].template get<string>();
   string mesh_file = options["mesh"]["file"].template get<string>();
   string tess_file = options["tess-file-old"].template get<string>();
   
   cout << model_file_old << endl;


   try
   {
      LEAnalogySolver solver(opt_file_name);
      std::cout << "Solving..." << std::endl;
      solver.solveForState();
      std::cout << "Solver Done" << std::endl;

   }
   catch (MISOException &exception)
   {

   }
#ifdef MFEM_USE_MPI
   MPI_Finalize();
#endif
}
#endif
