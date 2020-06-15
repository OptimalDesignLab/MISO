// Solve for the steady flow around a NACA0012

// set this const expression to true in order to use entropy variables for state
constexpr bool entvar = false;

#include<random>
#include <fstream>
#include <iostream>

#include "mfem.hpp"

#include "euler.hpp"

using namespace std;
using namespace mfem;
using namespace mach;

std::default_random_engine gen(std::random_device{}());
std::uniform_real_distribution<double> normal_rand(-1.0,1.0);

/// \brief Defines the random function for the jabocian check
/// \param[in] x - coordinate of the point at which the state is needed
/// \param[out] u - conservative variables stored as a 4-vector
void pert(const Vector &x, Vector& p);

void uinit(const mfem::Vector &x, mfem::Vector &u);

int main(int argc, char *argv[])
{
   const char *options_file = "airfoil_steady_options.json";
#ifdef MFEM_USE_MPI
   // Initialize MPI if parallel
   MPI_Init(&argc, &argv);
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

   string opt_file_name(options_file);
   nlohmann::json options;
   nlohmann::json file_options;
   ifstream opts(opt_file_name);
   opts >> file_options;
   options.merge_patch(file_options);

   try
   {
      // construct the solver, set the initial condition, and solve
      string opt_file_name(options_file);
      std::unique_ptr<mfem::Mesh> smesh;
      smesh.reset(new mfem::Mesh(options["mesh"]["file"].get<string>().c_str()));

      unique_ptr<AbstractSolver> solver(
         new EulerSolver<2, entvar>(opt_file_name, move(smesh)));
      // solver->initDerived();
      // Vector qfar(4);
      // static_cast<EulerSolver<2, entvar>*>(solver.get())->getFreeStreamState(qfar);
      // //Vector wfar(4);
      // // TODO: I do not like that we have to perform this conversion outside the solver...
      // //calcEntropyVars<double, 2>(qfar.GetData(), wfar.GetData());
      // solver->setInitialCondition(qfar);
      // solver->printSolution("airfoil-steady-init");
      // solver->checkJacobian(pert);
      // mfem::out << "\ninitial residual norm = " << solver->calcResidualNorm()
      //           << endl;
      // solver->solveForState();
      // solver->printSolution("airfoil-steady-final");
      // mfem::out << "\nfinal residual norm = " << solver->calcResidualNorm()
      //           << endl;
      // static_cast<EulerSolver<2, entvar>*>(solver.get())->verifyParamSens();
      solver->calcStatistics();
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

// perturbation function used to check the jacobian in each iteration
void pert(const Vector &x, Vector& p)
{
   p.SetSize(4);
   for (int i = 0; i < 4; i++)
   {
      p(i) = normal_rand(gen);
   }
}

/// Start with 0 velocity
void uinit(const mfem::Vector &x, mfem::Vector &q)
{
   Vector u;
   u.SetSize(x.Size()+2);
   q.SetSize(x.Size()+2);

   double rho = 1.0;

   u(0) = rho;
   if (x.Size() == 1)
   {
      u(1) = 0; // ignore angle of attack
   }
   else
   {
      u(1) = 0;
      u(2) = 0;
   }
   u(x.Size()+1) = 1/(euler::gamma*euler::gami);

   if (entvar == true)
   {
      calcEntropyVars<double, 2>(u.GetData(), q.GetData());
   }
   else
   {
      q = u;
   }
}