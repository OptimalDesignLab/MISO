// Using this to test different ideas

#include "mfem.hpp"
#include "advection.hpp"
#include <fstream>
#include <iostream>
#include "galer_diff.hpp"

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
   const char *options_file = "mach_options.json";
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
      mfem::GalerkinDifference gd(opt_file_name);
      DenseMatrix nmat1, nmat2;
      gd.BuildNeighbourMat(nmat1, nmat2);
      std::vector<int> nels;
      mfem::Vector cent;
      int degree = 2;
      int req_n = ((degree + 1) * (degree + 2)) / 2;
      gd.GetNeighbourSet(0, req_n, nels );
      cout << "Patch elements of element " << id << " : " << endl;
      for (int i = 0; i < nels.size(); ++i)
      {
         cout << nels[i] << endl;
      }
      gd.GetElementCenter(0, cent);
      cout << cent[0] << " , " << cent[1] << " , " << cent[2] << " " << endl;
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

