/// Solve the unsteady isentropic vortex problem
// set this const expression to true in order to use entropy variables for state
constexpr bool entvar = false;
#include<random>
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
void u0_function(const Vector &x, Vector& u0);

int main(int argc, char *argv[])
{
   const char *options_file = "unsteady_vortex_options.json";
#ifdef MFEM_USE_PETSC
   const char *petscrc_file = "eulersteady.petsc";
   // Get the option file
   nlohmann::json options;
   ifstream option_source(options_file);
   option_source >> options;
   // Write the petsc option file
   ofstream petscoptions(petscrc_file);
   const string linearsolver_name = options["petscsolver"]["ksptype"].get<string>();
   const string prec_name = options["petscsolver"]["pctype"].get<string>();
   petscoptions << "-solver_ksp_type " << linearsolver_name << '\n';
   petscoptions << "-prec_pc_type " << prec_name << '\n';
   //petscoptions << "-prec_pc_factor_levels " << 4 << '\n';
   petscoptions.close();
#endif
#ifdef MFEM_USE_MPI
   // Initialize MPI if parallel
   MPI_Init(&argc, &argv);
#endif
#ifdef MFEM_USE_PETSC
   MFEMInitializePetsc(NULL, NULL, petscrc_file, NULL);
#endif
   // Parse command-line options
   OptionsParser args(argc, argv);
   
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
         new EulerSolver<2, entvar>(opt_file_name, nullptr));
      solver->feedpert(pert);
      solver->initDerived();
      solver->setInitialCondition(u0_function);
      mfem::out << "\n|| u_h - u ||_{L^2} = " 
                << solver->calcL2Error(u0_function) << '\n' << endl;      
      solver->solveForState();
      mfem::out << "\n|| u_h - u ||_{L^2} = " 
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
#ifdef MFEM_USE_PETSC
   MFEMFinalizePetsc();
#endif
#ifdef MFEM_USE_MPI
   MPI_Finalize();
#endif
}

// Initial condition; see Crean et al. 2018 for the notation
void u0_function(const Vector &x, Vector& q)
{
   q.SetSize(4);
   Vector u0(4);
   double t = 0.0; // this could be an input...
   double x0 = 0.5;
   double y0 = 0.5;
   // scale is used to reduce the size of the vortex; need to scale the
   // velocity and pressure carefully to account for this
   double scale = 15.0;
   double xi = (x(0) - x0)*scale - t;
   double eta = (x(1) - y0)*scale;
   double M = 0.5;
   double epsilon = 1.0;
   double f = 1.0 - (xi*xi + eta*eta);
   // density
   u0(0) = pow(1.0 - epsilon*epsilon*euler::gami*M*M*exp(f)/(8*M_PI*M_PI),
               1.0/euler::gami);
   // x velocity
   u0(1) = 1.0 - epsilon*eta*exp(f*0.5)/(2*M_PI);
   u0(1) *= scale*u0(0);
   // y velocity
   u0(2) = epsilon*xi*exp(f*0.5)/(2*M_PI);
   u0(2) *= scale*u0(0);
   // pressure, used to get the energy
   double press = pow(u0(0), euler::gamma)/(euler::gamma*M*M);
   press *= scale*scale;
   u0(3) = press/euler::gami + 0.5*(u0(1)*u0(1) + u0(2)*u0(2))/u0(0);
   if (entvar == false)
   {
      q = u0;
   }
   else
   {
      calcEntropyVars<double, 2>(u0.GetData(), q.GetData());
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