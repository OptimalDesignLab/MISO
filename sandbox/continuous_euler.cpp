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
      mfem::out << "\n|| u_h - u ||_{L^2} = "
                << solver->calcL2Error(u0_function) << '\n' << endl;
      // solver->solveForState();
      // solver->PrintSodShock("sod_shock_final");
      // solver->PrintSodShockCenter("sod_shock_center_final");
      // mfem::out << "\n max(rho_h - rho) = " 
      //           << solver->calcSodShockMaxError(u1_function, 0) << '\n' << endl;
      // mfem::out << "\n|| rho_h - rho ||_{L^1} = " 
      //           << solver->calcSodShockL1Error(u1_function, 0) << '\n' << endl;
      // mfem::out << "\n|| u_h - u ||_{L^2} = " 
      //           << solver->calcL2Error(u1_function) << '\n' << endl;
      // double entropy = solver->calcOutput("entropy");
      // mfem::out << "entropy is " << entropy << '\n';
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


// Final condition at t = 0.3
void u1_function(const Vector &x, Vector &q)
{
   double t_final = 0.3;
   double xd = 0.5;
   // initial condition
   double rhol = 5.0;
   double rhor = 0.5;
   double ul   = 0.0;
   double ur   = 0.0;
   double pl   = 1.0;
   double pr   = 0.1;
   // some intermediate variables
   double cl = pow(euler::gamma * pl / rhol, 0.5);
   double cr = pow(euler::gamma * pr / rhor, 0.5);
   double alpha = (euler::gamma + 1.0) / (euler::gamma - 1.0);
   auto P_relation = [&] (double f)
   {
      double a = (ul - ur) / cr;
      double b = 2.0 / euler::gami * (cl / cr) * pow(1.0 - f * pr / pl, euler::gami / (2.0 * euler::gamma));
      double c = (f - 1.0) / pow(1.0+alpha*f, 0.5) * pow(2.0/euler::gamma/euler::gami, 0.5);
      return a + b - c;
   };
   //double f = bisection(P_relation, 0.0, 10.0, 1e-13, 1e-13, 100);
   double f = 2.848160188557576; // hard coded, need to be changed
   double p2 = f * pr;
   double p3 = p2;
   double rho2 = rhor * (1.0 + alpha * f)/(alpha + f);
   double rho3 = rhol * pow(p3/pl, 1.0/euler::gamma);
   double u2 = ur + cr * ((f -1.0)/pow(1.0 + alpha * f, 0.5)) * pow(2.0/euler::gamma/euler::gami, 0.5);
   double u3 = u2;
   double c2 = pow(euler::gamma * p2 /rho2, 0.5);
   double c3 = pow(euler::gamma * p3 /rho3, 0.5);
   double V = u2;
   double C = ur + cr * cr * (f - 1.0) / (euler::gamma * (u2 - ur));

   // Starting points of each region: xl = [0 x5), x5 = [x5, x3), x3 = [x3, x2), x2 = [x3, xr), xr = [xr, 1]
   double x5 = t_final * (ul - cl) + xd; 
   double x3 = t_final * ((euler::gamma + 1.0) * V/2.0 - cl - euler::gami * ul / 2.0) + xd;
   double x2 = t_final * V + xd;
   double xr = t_final * C +xd;

   // compute the solution
   q.SetSize(3); // without pressure
   Vector u0(3);
   if (x(0) < x5)
   {
      // region xl \in [0, x5)
      u0(0) = rhol;
      u0(1) = rhol * ul;
      u0(2) = pl/euler::gami + 0.5 * rhol * ul * ul;
   }
   else if (x5 <= x(0) && x(0) < x3)
   {
      // region x \in [x5, x3)
      double u5 = 2.0 / (euler::gamma + 1.0) * ( (x(0) - xd)/t_final + cl + euler::gami * ul / 2.0);
      double c5 = u5 - (x(0) - xd)/t_final;
      double p5 = pl * pow(c5/cl, 2.0 * euler::gamma / euler::gami);
      double rho5 = euler::gamma * p5 / pow(c5, 2.0);
      u0(0) = rho5;
      u0(1) = rho5 * u5;
      u0(2) = p5/euler::gami + 0.5 * rho5 * u5 * u5;
   }
   else if (x3 <= x(0) && x(0) < x2)
   {
      // region x \in [x3, x2)
      u0(0) = rho3;
      u0(1) = rho3 * u3;
      u0(2) = p3/euler::gami + 0.5 * rho3 * u3 * u3;
   }
   else if (x2 <= x(0) && x(0) < xr)
   {
      // region x \in [x3, x2)
      u0(0) = rho2;
      u0(1) = rho2 * u2;
      u0(2) = p2/euler::gami + 0.5 * rho2 * u2 * u2;
   }
   else
   {
      //region x \in [xr, 1]
      u0(0) = rhor;
      u0(1) = rhor * ur;
      u0(2) = pr/euler::gami + 0.5 * rhor * ur * ur;
   }

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
