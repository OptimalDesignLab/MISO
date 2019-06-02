// Using this to test different ideas

#include "mfem.hpp"
#include "advection.hpp"
#include <fstream>
#include <iostream>

using namespace std;
using namespace mfem;
using namespace mach;

/*!
* \brief Defines the velocity field
* \param[in] x - coordinate of the point at which the velocity is needed
* \param[out] v - velocity components at \a x
*/
void velocity_function(const Vector &x, Vector &v);

/*!
* \brief Defines the initial condition
* \param[in] x - coordinate of the point at which the velocity is needed
* \param[out] u0 - scalar initial condition stored as a 1-vector
*/
void u0_function(const Vector &x, Vector& u0);

int main(int argc, char *argv[])
{
   // Parse command-line options
   OptionsParser args(argc, argv);

   try
   {
      // construct the solver and set the initial condition
      AdvectionSolver solver(args, velocity_function);
      solver.set_initial_condition(u0_function);
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

// Initial condition
void u0_function(const Vector &x, Vector& u0)
{
   u0.SetSize(1);
   double r2 = pow(x(0) - 0.5, 2.0) + pow(x(1) - 0.5, 2.0);
   r2 *= 4.0;
   if (r2 > 1.0)
   {
      u0(0) = 1.0;
   } 
   else
   {
      // the following is an expansion of u = 1.0 - (r2 - 1.0).^5
      u0(0) = 2 - 5*r2 + 10*pow(r2,2) - 10*pow(r2,3) + 5*pow(r2,4) - pow(r2,5);
   }
}
