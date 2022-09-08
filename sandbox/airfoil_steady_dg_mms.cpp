/// Solve the steady isentropic vortex problem on a quarter annulus

// set this const expression to true in order to use entropy variables for state
constexpr bool entvar = false;

#include <random>
#include "adept.h"

#include "mfem.hpp"
#include "euler_dg.hpp"
#include <fstream>
#include <iostream>

using namespace std;
using namespace mfem;
using namespace mach;

std::default_random_engine gen(std::random_device{}());
std::uniform_real_distribution<double> normal_rand(-1.0, 1.0);

/// \brief Defines the random function for the jabocian check
/// \param[in] x - coordinate of the point at which the state is needed
/// \param[out] u - conservative variables stored as a 4-vector
void pert(const Vector &x, Vector &p);

/// \brief Returns the value of the integrated math entropy over the domain
double calcEntropyTotalExact();

/// \brief Defines the exact solution for the steady isentropic vortex
/// \param[in] x - coordinate of the point at which the state is needed
/// \param[out] u - state variables stored as a 4-vector
void uexact(const Vector &x, Vector &u);

/// Generate smoothly perturbed mesh 
/// \param[in] degree - polynomial degree of the mapping
/// \param[in] num_x - number of nodes in the x direction
/// \param[in] num_y - number of nodes in the y direction
Mesh buildCurvilinearMesh(int degree, int num_x, int num_y);

/// Generate square mesh
/// \param[in] N- number of elements in one direction
Mesh buildMesh(int N);
int main(int argc, char *argv[])
{
   const char *options_file = "airfoil_steady_dg_mms_options.json";
   // Initialize MPI
   int num_procs, rank;
   MPI_Init(&argc, &argv);
   MPI_Comm_size(MPI_COMM_WORLD, &num_procs);
   MPI_Comm_rank(MPI_COMM_WORLD, &rank);
   ostream *out = getOutStream(rank);
   int degree = 2.0;
   int nx = 20;
   int ny = 2;
   int N = 4;
   // Parse command-line options
   OptionsParser args(argc, argv);
   args.AddOption(&options_file, "-o", "--options", "Options file to use.");
   args.AddOption(&degree, "-d", "--degree", "poly. degree of mesh mapping");
   args.AddOption(&nx, "-nx", "--num-x", "number of x-direction segments");
   args.AddOption(&ny, "-ny", "--num-y", "number of y-direction segments");
   args.AddOption(&N, "-n", "--#elements", "number of elements");
   args.Parse();
   if (!args.Good())
   {
      args.PrintUsage(*out);
      return 1;
   }

   try
   {

      string opt_file_name(options_file);
      // construct the mesh
      unique_ptr<Mesh> smesh(new Mesh(buildMesh(N)));
      //unique_ptr<Mesh> smesh(new Mesh(buildCurvilinearMesh(degree, nx, ny)));
      // construct the solver and set initial conditions
      auto solver = createSolver<EulerDGSolver<2, entvar>>(opt_file_name, move(smesh));
      solver->setInitialCondition(uexact);
      solver->printSolution("airfoil_dg_mms_init", 0);

      // get the initial density error
      double l2_error = (static_cast<EulerDGSolver<2, entvar> &>(*solver)
                             .calcConservativeVarsL2Error(uexact, 0));
      double res_error = solver->calcResidualNorm();
      *out << "\n|| rho_h - rho ||_{L^2} = " << l2_error;
      *out << "\ninitial residual norm = " << res_error << endl;
      //solver->printResidual("residual-dg-mms-init", 0);
      // solver->checkJacobian(pert);
      solver->solveForState();
      solver->printSolution("airfoil_dg_mms_final", 0);
      // get the final density error
      l2_error = (static_cast<EulerDGSolver<2, entvar> &>(*solver)
                      .calcConservativeVarsL2Error(uexact, 0));
      res_error = solver->calcResidualNorm();
      auto drag_opts = R"({ "boundaries": [1, 1, 1, 1]})"_json;
      solver->createOutput("drag", drag_opts);
      double drag = abs(solver->calcOutput("drag"));
      // double entropy = solver->calcOutput("entropy");
      out->precision(15);
      *out << "\nfinal residual norm = " << res_error;
      *out << "\n|| rho_h - rho ||_{L^2} = " << l2_error << endl;
      *out << "\nDrag error = " << drag << endl;
      // *out << "\nTotal entropy = " << entropy;
      // *out << "\nEntropy error = "
      //      << fabs(entropy - calcEntropyTotalExact()) << endl;
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
void pert(const Vector &x, Vector &p)
{
   p.SetSize(4);
   for (int i = 0; i < 4; i++)
   {
      p(i) = normal_rand(gen);
   }
}

// Returns the exact total entropy value over the quarter annulus
// Note: the number 8.74655... that appears below is the integral of r*rho over
// the radii from 1 to 3.  It was approixmated using a degree 51 Gaussian
// quadrature.
double calcEntropyTotalExact()
{
   double rhoi = 2.0;
   double prsi = 1.0 / euler::gamma;
   double si = log(prsi / pow(rhoi, euler::gamma));
   return -si * 8.746553803443305 * M_PI * 0.5 / 0.4;
}

/// Exact solution
void uexact(const Vector &x, Vector &q)
{
   q.SetSize(4);
   Vector u(4);
   const double rho0 = 1.0;
   const double rhop = 0.05;
   const double u0 = 0.5;
   const double up = 0.05;
   const double T0 = 1.0;
   const double Tp = 0.05;
   const double scale = 1.0;
   const double trans = 0.0;
   /// define the exact solution
   double rho = rho0 + rhop * pow(sin(M_PI * (x(0) + trans) / scale), 2) *
                           sin(M_PI * (x(1) + trans) / scale);
   double ux =
       4.0 * u0 * ((x(1) + trans) / scale) * (1.0 - (x(1) + trans)/ scale) +
       (up * sin(2.0 * M_PI * ((x(1) + trans)) / scale) * pow(sin(M_PI * (x(0)+ trans) / scale), 2));
   double uy =
       -up * pow(sin(2.0 * M_PI * (x(0) + trans) / scale), 2) * sin(M_PI * (x(1) + trans) / scale);
   double T = T0 + Tp * (pow((x(0) + trans) / scale, 4) - (2.0 * pow((x(0) + trans) / scale, 3)) +
                         pow((x(0) + trans) / scale, 2) + pow((x(1) + trans) / scale, 4) -
                         (2.0 * pow((x(1) + trans) / scale, 3)) + pow((x(1) + trans) / scale, 2));
   double p = rho * T;
   double e = (p / (euler::gamma - 1)) + 0.5 * rho * (ux * ux + uy * uy);
   u(0) = rho;
   u(1) = rho*ux;  // multiply by rho ?
   u(2) = rho*uy;
   u(3) = e;
   if (entvar == false)
   {
      q = u;
   }
   else
   {
      calcEntropyVars<double, 2>(u.GetData(), q.GetData());
   }
}

Mesh buildMesh(int N)
{
   Mesh mesh = Mesh::MakeCartesian2D(N, N, Element::QUADRILATERAL,
                                     true /* gen. edges */, 1.0, 1.0,
                                     true);
   return mesh;
}

Mesh buildCurvilinearMesh(int degree, int num_x, int num_y)
{
   Mesh mesh = Mesh::MakeCartesian2D(num_x, num_y, Element::QUADRILATERAL,
                                     true /* gen. edges */, 1.0, 1.0, true);
   // strategy:
   // 1) generate a fes for Lagrange elements of desired degree
   // 2) create a Grid Function using a VectorFunctionCoefficient
   // 4) use mesh_ptr->NewNodes(nodes, true) to set the mesh nodes
   
   // Problem: fes does not own fec, which is generated in this function's scope
   // Solution: the grid function can own both the fec and fes
   H1_FECollection *fec = new H1_FECollection(degree, 2 /* = dim */);
   FiniteElementSpace *fes = new FiniteElementSpace(&mesh, fec, 2,
                                                    Ordering::byVDIM);

   auto xy_fun = [](const Vector& xi, Vector &x)
   {
      x(0) = xi(0) + (1.0/40.0)*sin(2.0*M_PI*xi(0))*sin(2.0*M_PI*xi(1));
      x(1) = xi(1) + (1.0/40.0)*sin(2.0*M_PI*xi(1))*sin(2.0*M_PI*xi(0));
   };
   VectorFunctionCoefficient xy_coeff(2, xy_fun);
   GridFunction *xy = new GridFunction(fes);
   xy->MakeOwner(fec);
   xy->ProjectCoefficient(xy_coeff);
   mesh.NewNodes(*xy, true);
   return mesh;
}