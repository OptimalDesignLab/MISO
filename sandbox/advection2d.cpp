// Using this to test different ideas

#include "mfem.hpp"
#include "advection.hpp"
#include <fstream>
#include <iostream>

using namespace std;
using namespace mach;

/*!
* \brief Defines the velocity field
* \param[in] x - coordinate of the point at which the velocity is needed
* \param[out] v - velocity components at \a x
*/
void velocity_function(const Vector &x, Vector &v);

int main(int argc, char *argv[])
{
   // Parse command-line options
   OptionsParser args(argc, argv);

   try
   {
      // Attempt to construct the solver
      AdvectionSolver solver(args, velocity_function);
   }
   catch (MachException &exception)
   {
      exception.print_message();
   }
   catch (std::exception &exception)
   {
      cerr << exception.what() << endl;
   }
}

void velocity_function(const Vector &x, Vector &v)
{
   // Simply advection to upper right corner; See mfem ex9 to see how this might
   // be generalized.
   v(0) = 1.0;
   v(1) = 1.0;
}
