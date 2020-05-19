constexpr bool entvar = false;

#include<random>

#include "mfem.hpp"
#include "euler.hpp"
#include "euler_integ.hpp"
#include "sbp_fe.hpp"

#include <fstream>
#include <iostream>

using namespace std;
using namespace mfem;
using namespace mach;

std::default_random_engine gen(std::random_device{}());
std::uniform_real_distribution<double> normal_rand(-1.0,1.0);
void uinit(const mfem::Vector &x, mfem::Vector &u);
void pert(const Vector &x, Vector& p);

static double aoa_s = 0.0;
static double mach_s = 0.0;
static double iroll = 0.0;
static double ipitch = 0.0;

int main(int argc, char *argv[])
{
   ostream *out;
   // Parse command-line options
   OptionsParser args(argc, argv);
   const char *options_file = "naca_stoch_test_options.json";
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
#ifdef MFEM_USE_PETSC
   const char *petscrc_file = "eulersteady.petsc";
   // Get the option file
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
   int rank;
   MPI_Comm_rank(MPI_COMM_WORLD, &rank);
   out = getOutStream(rank); 
#else
   out = getOutStream(0);
#endif
#ifdef MFEM_USE_PETSC
   MFEMInitializePetsc(NULL, NULL, petscrc_file, NULL);
#endif


   int degree = options["space-dis"]["degree"].get<int>();
   aoa_s = options["flow-param"]["aoa"].template get<double>()*M_PI/180;
   mach_s = options["flow-param"]["mach"].template get<double>();
   iroll = options["flow-param"]["roll-axis"].template get<int>();
   ipitch = options["flow-param"]["pitch-axis"].template get<int>();
   std::unique_ptr<mfem::Mesh> smesh;
   smesh.reset(new mfem::Mesh(options["mesh"]["file"].get<string>().c_str()));

   try
   {
      std::cout <<"Number of elements " << smesh->GetNE() <<'\n';
      ofstream mesh_ofs("naca_stoch_test.vtk");
      mesh_ofs.precision(8);
      smesh->PrintVTK(mesh_ofs);

      unique_ptr<AbstractSolver> solver(new EulerSolver<2, entvar>(opt_file_name, move(smesh)));
      solver->initDerived();

      solver->setInitialCondition(uinit);
      double res_error = solver->calcResidualNorm();
      if (0==rank)
      {
         mfem::out << "\ninitial residual norm = " << res_error << endl;
      }
      solver->printSolution("naca_init", degree+1);
      //solver->checkJacobian(pert);
      solver->solveForState();
      res_error = solver->calcResidualNorm();
      solver->printSolution("naca_final",degree+1);

      if (0==rank)
      {
         mfem::out << "\nfinal residual norm = " << res_error;
      }

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

/// Start with uniform flow? At the proper angle of attack
void uinit(const mfem::Vector &x, mfem::Vector &q)
{
   Vector u;
   u.SetSize(x.Size()+2);
   q.SetSize(x.Size()+2);

   double rho = 1.0;

   u(0) = rho;
   if (x.Size() == 1)
   {
      u(1) = u(0)*mach_s; // ignore angle of attack
   }
   else
   {
      u(iroll+1) = u(0)*mach_s*cos(aoa_s);
      u(ipitch+1) = u(0)*mach_s*sin(aoa_s);
   }
   u(x.Size()+1) = 1/(euler::gamma*euler::gami) + 0.5*mach_s*mach_s;

   if (entvar == true)
   {
      calcEntropyVars<double, 2>(u.GetData(), q.GetData());
   }
   else
   {
      q = u;
   }
}

void pert(const Vector &x, Vector& p)
{
   p.SetSize(x.Size()+2);
   for (int i = 0; i < 4; i++)
   {
      p(i) = normal_rand(gen);
   }
}