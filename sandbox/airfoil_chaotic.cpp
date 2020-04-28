// Solve for the flow around a NACA0012 at high angle of attack

#include <fstream>
#include <iostream>

#include "mfem.hpp"

#include "mach.hpp"

using namespace std;
using namespace mfem;
using namespace mach;

void uinit(const Vector &x, Vector& u0);

int main(int argc, char *argv[])
{
   const char *options_file = "airfoil_chaotic_options.json";
#ifdef MFEM_USE_PETSC
   const char *petscrc_file = "airfoil_chaotic.petsc";
   //Get the option files
   nlohmann::json options;
   ifstream option_source(options_file);
   option_source >> options;
   // write the petsc linear solver options from options
   ofstream petscoptions(petscrc_file);
   const string linearsolver_name = options["petscsolver"]["ksptype"].get<string>();
   const string prec_name = options["petscsolver"]["pctype"].get<string>();
   petscoptions << "-solver_ksp_type " << linearsolver_name << endl;
   petscoptions << "-prec_pc_type " << prec_name << endl;
   petscoptions << "-pc_factor_mat_ordering_type rcm" << endl;
   petscoptions << "-pc_factor_levels 0" << endl;
   petscoptions << "-pc_factor_reuse_ordering" << endl;
   petscoptions.close();
#endif
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
#ifdef MFEM_USE_PETSC
   MFEMInitializePetsc(NULL, NULL, petscrc_file, NULL);
#endif

   try
   {
      // construct the solver, set the initial condition, and solve
      string opt_file_name(options_file);
#if 1
      unique_ptr<NavierStokesSolver<2>> solver(
         new NavierStokesSolver<2>(opt_file_name, nullptr));
#endif
#if 0
      unique_ptr<EulerSolver<2>> solver(
         new EulerSolver<2>(opt_file_name, nullptr));
#endif
      solver->initDerived();
      Vector qfar(4);
      solver->getFreeStreamState(qfar);

      // TEMP?
      qfar(1) = 0.0;
      qfar(2) = 0.0;

      //static_cast<EulerSolver<2>*>(solver)->getFreeStreamState(qfar); 
      solver->setInitialCondition(qfar);
      solver->printSolution("airfoil-chaotic-init");

      mfem::out << "\ninitial residual norm = " << solver->calcResidualNorm()
                << endl;

      solver->solveForState();
      solver->printSolution("airfoil-chaotic-final");

      mfem::out << "\nfinal residual norm = " << solver->calcResidualNorm()
                << endl;
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

void uinit(const Vector &x, Vector& u0)
{
   u0.SetSize(4);
   u0 = 1.0;
}
