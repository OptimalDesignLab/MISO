/// Solve the unsteady isentropic vortex problem
// set this const expression to true in order to use entropy variables for state
constexpr bool entvar = false;
#include <random>
#include "mfem.hpp"
#include "euler.hpp"
#include "euler_fluxes.hpp"
#include <fstream>
#include <iostream>

using namespace std;
using namespace mfem;
using namespace mach;

std::default_random_engine gen(std::random_device{}());
std::uniform_real_distribution<double> normal_rand(-1.0,1.0);
/// \brief Defines the random function for the jabocian check
/// \param[in] x - coordinate of the point at which the state is needed
/// \param[out] u - conservative variables stored as a 4-vector
void pert(const Vector &x, Vector& p);

/// \brief Defines the initial condition for the unsteady isentropic vortex
/// \param[in] x - coordinate of the point at which the state is needed
/// \param[out] u0 - conservative variables stored as a 4-vector
void u0_function(const Vector &x, Vector &u0);
void u1_function(const Vector &x, Vector &u0);

int main(int argc, char *argv[])
{
   const char *options_file = "continuous_euler_options.json";
   // Parse command-line options
   OptionsParser args(argc, argv);
   int nx = 50;
   args.AddOption(&nx, "-n", "--number", "Number of elements");
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
         new EulerSolver<1, entvar>(opt_file_name, nullptr));
      // unique_ptr<Mesh> mesh(new Mesh(nx));
      // unique_ptr<AbstractSolver> solver(
      //    new EulerSolver<1, entvar>(opt_file_name, move(mesh)));
      solver->feedpert(pert);
      solver->initDerived();
      solver->setInitialCondition(u0_function);
      mfem::out << "After set initial condition.\n";
      solver->feedpert(pert);
      mfem::out << "after peet perturbation.\n";
      solver->PrintSodShock("continuous_euler_init");
      solver->PrintSodShockCenter("continuous_euler_center_init");
      mfem::out << "\n max(rho_h - rho) = " 
                << solver->calcSodShockMaxError(u0_function, 0) << '\n' << endl;
      mfem::out << "\n|| rho_h - rho ||_{L^1} = " 
                << solver->calcSodShockL1Error(u0_function, 0) << '\n' << endl;
      mfem::out << "\n|| rho_h - rho ||_{L^2} = "
                << solver->calcL2Error(u0_function) << '\n' << endl;
      solver->solveForState();
      solver->PrintSodShock("continuous_euler_final");
      solver->PrintSodShockCenter("continuous_euler_center_final");
      mfem::out << "\n max(rho_h - rho) = " 
                << solver->calcSodShockMaxError(u1_function, 0) << '\n' << endl;
      mfem::out << "\n|| rho_h - rho ||_{L^1} = " 
                << solver->calcSodShockL1Error(u1_function, 0) << '\n' << endl;
      mfem::out << "\n|| rho_h - rho ||_{L^2} = " 
                << solver->calcL2Error(u1_function,0) << '\n' << endl;
      double entropy = solver->calcOutput("entropy");
      mfem::out << "entropy is " << entropy << '\n';
   }
   catch (MachException &exception)
   {
      exception.print_message();
   }
   catch (std::exception &exception)
   {
      cerr << exception.what() << endl;
   }
   return 0;
}

// Initial condition; see Ranocho et al. 2019 for the notation
void u0_function(const Vector &x, Vector& q)
{
   q.SetSize(3);
   Vector u0(3);
   u0(0) = 1.0 + 0.98*sin(2.0 * M_PI * x(0));
   u0(1) = u0(0)*0.1;
   u0(2) = 20./euler::gami + 0.5 * u0(0) * 0.01;
   if (entvar == false)
   {
      q = u0;
   }
   else
   {
      calcEntropyVars<double, 1>(u0.GetData(), q.GetData());
   }
}

// Initial condition; see Ranocho et al. 2019 for the notation
// void u0_function(const Vector &x, Vector& q)
// {
//    q.SetSize(3);
//    Vector u0(3);
//    u0(0) = 1.0;
//    u0(1) = 3.0 *x(0) - 6.0;
//    u0(2) = x(0) * x(0) + 7.0 * x(0) - 10.0;
//    if (entvar == false)
//    {
//       q = u0;
//    }
//    else
//    {
//       calcEntropyVars<double, 1>(u0.GetData(), q.GetData());
//    }
// }


// Final condition at t = 0.3
void u1_function(const Vector &x, Vector &q)
{
   double t_final = 1.0;
   // compute the solution
   q.SetSize(3); // without pressure
   Vector u0(3);
   u0(0) = 1.0 + 0.98 * sin(2.*M_PI * (x(0) - 0.1 * t_final));
   u0(1) = 0.1*u0(0);
   u0(2) = 20./euler::gami + 0.5 * u0(0) * 0.01;

   if (entvar == false)
   {
      q = u0;
   }
   else
   {
      calcEntropyVars<double, 1>(u0.GetData(), q.GetData());
   }
}
// perturbation function used to check the jacobian in each iteration
void pert(const Vector &x, Vector& p)
{
   p.SetSize(4);
   for (int i = 0; i < 4; i++)
   {
      p(i) = normal_rand(gen);
   }
}
