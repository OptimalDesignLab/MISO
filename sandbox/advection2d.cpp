// Using this to test different ideas

#include "mfem.hpp"
#include "advection.hpp"
#include <fstream>
#include <iostream>

using namespace std;
using namespace mfem;
using namespace mach;

/// Defines the velocity field
/// \param[in] x - coordinate of the point at which the velocity is needed
/// \param[out] v - velocity components at \a x
void velocity_function(const Vector &x, Vector &v);

/// \brief Defines the initial condition
/// \param[in] x - coordinate of the point at which the velocity is needed
/// \param[out] u0 - scalar initial condition stored as a 1-vector
void u0_function(const Vector &x, Vector& u0);

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
      unique_ptr<AbstractSolver> solver(
         new AdvectionSolver<2>(opt_file_name, velocity_function));
      solver->setInitialCondition(u0_function);
      *out << "\n|| u_h - u ||_{L^2} = " 
                << solver->calcL2Error(u0_function) << '\n' << endl;      
      solver->solveForState();
      *out << "\n|| u_h - u ||_{L^2} = " 
                << solver->calcL2Error(u0_function) << '\n' << endl;
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

