// set this const expression to true in order to use entropy variables for state (doesn't work for rans)
constexpr bool entvar = false;

#include<random>
#include "adept.h"

#include "mfem.hpp"
#include "rans.hpp"
#include <fstream>
#include <iostream>

using namespace std;
using namespace mfem;
using namespace mach;

std::default_random_engine gen(std::random_device{}());
std::uniform_real_distribution<double> normal_rand(-1.0,1.0);

static double mu;
static double mach_fs;
static double aoa_fs;
static int iroll;
static int ipitch;
static double chi_fs;

/// \brief Defines the random function for the jacobian check
/// \param[in] x - coordinate of the point at which the state is needed
/// \param[out] u - conservative + SA variables stored as a 5-vector
void pert(const Vector &x, Vector& p);

/// \brief Defines the exact solution for the rans freestream problem
/// \param[in] x - coordinate of the point at which the state is needed
/// \param[out] u - conservative + SA variables stored as a 5-vector
void uexact(const Vector &x, Vector& u);

int main(int argc, char *argv[])
{
   const char *options_file = "rans_freestream_options.json";

   // Initialize MPI
   int num_procs, rank;
   MPI_Init(&argc, &argv);
   MPI_Comm_size(MPI_COMM_WORLD, &num_procs);
   MPI_Comm_rank(MPI_COMM_WORLD, &rank);
   ostream *out = getOutStream(rank);
  
   // Parse command-line options
   OptionsParser args(argc, argv);
   int degree = 2.0;
   int nx = 1;
   int ny = 1;
   args.AddOption(&nx, "-nx", "--numx",
                  "Number of elements in x direction");
   args.AddOption(&ny, "-ny", "--numy",
                  "Number of elements in y direction");
   args.Parse();
   if (!args.Good())
   {
      args.PrintUsage(*out);
      return 1;
   }

   try
   {
      // construct the mesh
      string opt_file_name(options_file);
      nlohmann::json file_options;
      std::ifstream opts(opt_file_name);
      opts >> file_options;
      mu = file_options["flow-param"]["mu"].template get<double>();
      mach_fs = file_options["flow-param"]["mach"].template get<double>();
      aoa_fs = file_options["flow-param"]["aoa"].template get<double>()*M_PI/180;
      iroll = file_options["flow-param"]["roll-axis"].template get<int>();
      ipitch = file_options["flow-param"]["pitch-axis"].template get<int>();
      chi_fs = file_options["flow-param"]["chi"].template get<double>();
      // generate a simple tri mesh
      std::unique_ptr<Mesh> smesh(new Mesh(nx, ny, 
                              Element::TRIANGLE, true /* gen. edges */, 1.0,
                              1.0, true));
      *out << "Number of elements " << smesh->GetNE() <<'\n';

      // construct the solver and set initial conditions
      auto solver = createSolver<RANavierStokesSolver<2, entvar>>(opt_file_name,
                                                         move(smesh));
      solver->setInitialCondition(uexact);
      solver->printSolution("rans_init", 0);

      // get the initial density error
      //double l2_error = (static_cast<RANavierStokesSolver<2, entvar>&>(*solver)
      //                      .calcConservativeVarsL2Error(uexact, 0));
      double res_error = solver->calcResidualNorm();
      //*out << "\n|| rho_h - rho ||_{L^2} = " << l2_error;
      *out << "\ninitial residual norm = " << res_error << endl;
      solver->checkJacobian(pert);
      solver->solveForState();
      solver->printSolution("rans_final",0);
      // get the final density error
      //l2_error = (static_cast<RANavierStokesSolver<2, entvar>&>(*solver)
      //                      .calcConservativeVarsL2Error(uexact, 0));
      res_error = solver->calcResidualNorm();

      *out << "\nfinal residual norm = " << res_error;
      //*out << "\n|| rho_h - rho ||_{L^2} = " << l2_error << endl;
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

   MPI_Finalize();
}

// perturbation function used to check the jacobian in each iteration
void pert(const Vector &x, Vector& p)
{
   p.SetSize(5);
   for (int i = 0; i < 5; i++)
   {
      p(i) = normal_rand(gen);
   }
}

// Exact solution; same as freestream bc
void uexact(const Vector &x, Vector& q)
{
   q.SetSize(5);
   Vector u(5);
   
   u = 0.0;
   u(0) = 1.0;
   u(1) = u(0)*mach_fs*cos(aoa_fs);
   u(2) = u(0)*mach_fs*sin(aoa_fs);
   u(3) = 1/(euler::gamma*euler::gami) + 0.5*mach_fs*mach_fs;
   u(4) = chi_fs*(mu/u(0));

   if (entvar == false)
   {
      q = u;
   }
   else
   {
      throw MachException("No entvar for this");
   }
}

