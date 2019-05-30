#include "solver.hpp"

namespace mach
{
    
PDESolver::PDESolver(OptionsParser &args)
{
   // references to options here
   args.AddOption(&degree, "-d", "--degree",
                  "Degree of the SBP operators.");
   args.Parse();
   if (!args.Good())
   {
      args.PrintUsage(cout);
      // TODO: throw an exception here; need some exception codes first
   }
   args.PrintOptions(cout);

   // This is just to make sure we are setting members correctly
   cout << "degree = " << degree << endl;
   //C_SBPCollection fec(order, dim);
}

} // namespace mach
