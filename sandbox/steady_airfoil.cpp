/// Solve the steady isentropic vortex problem on a quarter annulus

// set this const expression to true in order to use entropy variables for state
constexpr bool entvar = false;

#include<random>
#include "adept.h"

#include "mfem.hpp"
#include "euler.hpp"
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

/// \brief Returns the value of the integrated math entropy over the domain
double calcEntropyTotalExact();

/// \brief Defines the exact solution for the steady isentropic vortex
/// \param[in] x - coordinate of the point at which the state is needed
/// \param[out] u - state variables stored as a 4-vector
void uexact(const Vector &x, Vector& u);

/// Generate quarter annulus mesh 
/// \param[in] degree - polynomial degree of the mapping
/// \param[in] num_rad - number of nodes in the radial direction
/// \param[in] num_ang - number of nodes in the angular direction
Mesh buildQuarterAnnulusMesh(int degree, int num_rad, int num_ang);

int main(int argc, char *argv[])
{
   const char *options_file = "steady_airfoil_options.json";
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

   // Initialize MPI
   int num_procs, rank;
   MPI_Init(&argc, &argv);
   MPI_Comm_size(MPI_COMM_WORLD, &num_procs);
   MPI_Comm_rank(MPI_COMM_WORLD, &rank);
   ostream *out = getOutStream(rank);

#ifdef MFEM_USE_PETSC
   MFEMInitializePetsc(NULL, NULL, petscrc_file, NULL);
#endif
  
   // Parse command-line options
   OptionsParser args(argc, argv);
   int degree = 2;
   int nx = 1;
   int ny = 1;
   args.AddOption(&options_file, "-o", "--options",
                  "Options file to use.");
   args.AddOption(&degree, "-d", "--degree", "poly. degree of mesh mapping");
   args.AddOption(&nx, "-nr", "--num-rad", "number of radial segments");
   args.AddOption(&ny, "-nt", "--num-theta", "number of angular segments");
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
      const char *mesh_file = "airfoil_0012_tr_new_coarse_FF.mesh";
      auto smesh = unique_ptr<Mesh>(new Mesh(mesh_file, 1, 1));
      *out << "Number of elements " << smesh->GetNE() <<'\n';
      ofstream sol_ofs("steady_airfoil_mesh.vtk");
      sol_ofs.precision(14);
      smesh->PrintVTK(sol_ofs,0);

      // construct the solver and set initial conditions
      auto solver = createSolver<EulerSolver<2, entvar>>(opt_file_name,
                                                         move(smesh));
      solver->setInitialCondition(uexact);
      solver->printSolution("euler_init_airfoil", 0);

      // get the initial density error
      double l2_error = (static_cast<EulerSolver<2, entvar>&>(*solver)
                            .calcConservativeVarsL2Error(uexact, 0));
      double res_error = solver->calcResidualNorm();
      *out << "\n|| rho_h - rho ||_{L^2} = " << l2_error;
      *out << "\ninitial residual norm = " << res_error << endl;
      solver->checkJacobian(pert);
      solver->solveForState();
      solver->printSolution("euler_final_airfoil",0);
      // get the final density error
      l2_error = (static_cast<EulerSolver<2, entvar>&>(*solver)
                            .calcConservativeVarsL2Error(uexact, 0));
      res_error = solver->calcResidualNorm();
      double drag = abs(solver->calcOutput("drag"));
      double lift = abs(solver->calcOutput("lift"));
      double entropy = solver->calcOutput("entropy");

      out->precision(15);
      *out << "\nfinal residual norm = " << res_error;
      *out << "\n|| rho_h - rho ||_{L^2} = " << l2_error << endl;
      *out << "\nDrag error = " << drag << endl;
      *out << "\nLift is = " << lift << endl;
      *out << "\nTotal entropy = " << entropy;
      *out << "\nEntropy error = "
           << fabs(entropy - calcEntropyTotalExact()) << endl;
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
   p.SetSize(4);
   for (int i = 0; i < 4; i++)
   {
      p(i) = normal_rand(gen);
   }
}

// Returns the exact total entropy value over the quarter annulus
// Note: the number 8.74655... that appears below is the integral of r*rho over the radii
// from 1 to 3.  It was approixmated using a degree 51 Gaussian quadrature.
double calcEntropyTotalExact()
{
   double rhoi = 2.0;
   double prsi = 1.0/euler::gamma;
   double si = log(prsi/pow(rhoi, euler::gamma));
   return -si*8.746553803443305*M_PI*0.5/0.4;
}

// Exact solution; note that I reversed the flow direction to be clockwise, so
// the problem and mesh are consistent with the LPS paper (that is, because the
// triangles are subdivided from the quads using the opposite diagonal)
void uexact(const Vector &x, Vector& q)
{
   q.SetSize(4);
   double mach_fs = 0.5;
   double aoa = 3.0;
   double aoa_fs = aoa * M_PI / 180;
   Vector u(4);
   u(0) = 1.0;
   u(1) = u(0) * mach_fs * cos(aoa_fs);
   u(2) = u(0) * mach_fs * sin(aoa_fs);
   u(3) = 1 / (euler::gamma * euler::gami) + 0.5 * mach_fs * mach_fs;

   if (entvar == false)
   {
      q = u;
   }
   else
   {
      calcEntropyVars<double, 2>(u.GetData(), q.GetData());
   }
}

Mesh buildQuarterAnnulusMesh(int degree, int num_rad, int num_ang)
{
   Mesh mesh = Mesh::MakeCartesian2D(num_rad, num_ang, Element::TRIANGLE,
                                     true /* gen. edges */, 2.0, M_PI*0.5,
                                     true);
   // strategy:
   // 1) generate a fes for Lagrange elements of desired degree
   // 2) create a Grid Function using a VectorFunctionCoefficient
   // 4) use mesh_ptr->NewNodes(nodes, true) to set the mesh nodes
   
   // Problem: fes does not own fec, which is generated in this function's scope
   // Solution: the grid function can own both the fec and fes
   H1_FECollection *fec = new H1_FECollection(degree, 2 /* = dim */);
   FiniteElementSpace *fes = new FiniteElementSpace(&mesh, fec, 2,
                                                    Ordering::byVDIM);

   // This lambda function transforms from (r,\theta) space to (x,y) space
   auto xy_fun = [](const Vector& rt, Vector &xy)
   {
      xy(0) = (rt(0) + 1.0)*cos(rt(1)); // need + 1.0 to shift r away from origin
      xy(1) = (rt(0) + 1.0)*sin(rt(1));
   };
   VectorFunctionCoefficient xy_coeff(2, xy_fun);
   GridFunction *xy = new GridFunction(fes);
   xy->MakeOwner(fec);
   xy->ProjectCoefficient(xy_coeff);

   mesh.NewNodes(*xy, true);
   return mesh;
}
