// Note: this code needs to be optimized.
// it checks patch elements only upto degree two currently
// for 8 {0, 1, 2, 3, 8, 9, 10, 11} elements in tri320.smb.
// may have to switch to using switch statement for degree.

#include "mfem.hpp"
#include <fstream>
#include <iostream>
#include "galer_diff.hpp"

using namespace std;
using namespace mfem;
using namespace mach;

// function to match patch elements
void checkNeighbour(const std::string &, int);

int main(int argc, char *argv[])
{
#ifdef MFEM_USE_MPI
   // Initialize MPI if parallel
   MPI_Init(&argc, &argv);
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
   array<int, 2> degree;
   string opt_file_name(options_file);
   degree[0] = 1;
   degree[1] = 2;
   /// checks if the neighbours match with original ones
   for (int i=0; i < degree.size(); ++i)
   {
     checkNeighbour(opt_file_name, degree[i]);
   }
   
#ifdef MFEM_USE_MPI
   MPI_Finalize();
#endif
}
// this takes in options file name and degree
void checkNeighbour(const std::string &opt_file_name, int degree)
{
   // calculate required # of neighbours
   int req_n = ((degree + 1) * (degree + 2)) / 2;
   // this stores patch elements for all 8 elements considered
   std::vector<int> neigh;
   int k = 0;
   int id = 0;
   // arrays to store patch elements
   array<int, 28> d1_data;
   array<int, 62> d2_data;
   // degree 1 patch data for 8 elements
   d1_data = {0, 1, 3,
              1, 0, 8,
              2, 3, 5,
              3, 0, 2, 10,
              8, 1, 9, 11,
              9, 8, 16,
              10, 3, 11, 13,
              11, 8, 10, 18};
   // degree 2 patch data for 8 elements
   d2_data = {0, 1, 3, 8, 2, 10,
              1, 0, 8, 3, 9, 11,
              2, 3, 5, 0, 10, 4, 12,
              3, 0, 2, 10, 1, 5, 11, 13,
              8, 1, 9, 11, 0, 16, 10, 18,
              9, 8, 16, 1, 11, 17, 19,
              10, 3, 11, 13, 0, 2, 8, 18, 12, 20,
              11, 8, 10, 18, 1, 9, 3, 13, 19, 21};
   mfem::GalerkinDifference gd(opt_file_name);
   // get patch elements for all 8 elements
   for (int i = 0; i < 8; ++i)
   {
      std::vector<int> nels;
      // provides patch elements for element wth index 'id'
      gd.GetNeighbourSet(id, req_n, nels);
      for (int k = 0; k < nels.size(); ++k)
      {
         neigh.push_back(nels[k]);
      }
      // stride to go to next element in given degree patch data
      k += nels.size();
      if (degree == 1)
      {
         id = d1_data[k];
      }
      else
      {
         id = d2_data[k];
      }
   }
   // match patch elements
   if (degree == 1)
   {
      assert(neigh.size() == d1_data.size());
      for (int i = 0; i < d1_data.size(); ++i)
      {
         assert(neigh[i] == d1_data[i]);
      }
      cout << "-----------------------------" << endl;
      cout << "Patch test passed for degree " << degree << endl;
      cout << "-----------------------------" << endl;
      cout << endl;
   }
   else
   {
      assert(neigh.size() == d2_data.size());
      for (int i = 0; i < d2_data.size(); ++i)
      {
         assert(neigh[i] == d2_data[i]);
      }
      cout << "-----------------------------" << endl;
      cout << "Patch test passed for degree " << degree << endl;
      cout << "-----------------------------" << endl;
   }
}
