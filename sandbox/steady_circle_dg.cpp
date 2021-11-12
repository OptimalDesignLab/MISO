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
   const char *options_file = "steady_circle_dg_options.json";
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
    //   unique_ptr<Mesh> smesh(new Mesh(buildQuarterAnnulusMesh(degree, nx, ny)));
    //   *out << "Number of elements " << smesh->GetNE() <<'\n';
    //   ofstream sol_ofs("steady_circle_mesh_dg.vtk");
    //   sol_ofs.precision(14);
    //   smesh->PrintVTK(sol_ofs);

      // construct the solver and set initial conditions
      auto solver = createSolver<EulerSolver<2, entvar>>(opt_file_name);
      Vector qfar(4);
      static_cast<EulerSolver<2, entvar> *>(solver.get())
          ->getFreeStreamState(qfar);
      qfar.Print();
      solver->setInitialCondition(qfar);
      solver->printSolution("circle_init", 0);

      double res_error = solver->calcResidualNorm();
      *out << "\ninitial residual norm = " << res_error << endl;
      solver->checkJacobian(pert);
      solver->solveForState();
      solver->printSolution("circle_final",0);
      // get the final error
      res_error = solver->calcResidualNorm();
      auto drag_opts = R"({ "boundaries": [1, 0]})"_json;
      solver->createOutput("drag", drag_opts);
      double drag = abs(solver->calcOutput("drag"));
      // double entropy = solver->calcOutput("entropy");
      out->precision(15);
      *out << "\nfinal residual norm = " << res_error;
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
   Vector u(4);
   double ri = 1.0;
   double Mai = 0.5; //0.95 
   double rhoi = 2.0;
   double prsi = 1.0/euler::gamma;
   double rinv = ri/sqrt(x(0)*x(0) + x(1)*x(1));
   double rho = rhoi*pow(1.0 + 0.5*euler::gami*Mai*Mai*(1.0 - rinv*rinv),
                         1.0/euler::gami);
   double Ma = sqrt((2.0/euler::gami)*( ( pow(rhoi/rho, euler::gami) ) * 
                    (1.0 + 0.5*euler::gami*Mai*Mai) - 1.0 ) );
   double theta;
   if (x(0) > 1e-15)
   {
      theta = atan(x(1)/x(0));
   }
   else
   {
      theta = M_PI/2.0;
   }
   double press = prsi* pow( (1.0 + 0.5*euler::gami*Mai*Mai) / 
                 (1.0 + 0.5*euler::gami*Ma*Ma), euler::gamma/euler::gami);
   double a = sqrt(euler::gamma*press/rho);

   u(0) = rho;
   u(1) = -rho*a*Ma*sin(theta);
   u(2) = rho*a*Ma*cos(theta);
   u(3) = press/euler::gami + 0.5*rho*a*a*Ma*Ma;

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
   Mesh mesh = Mesh::MakeCartesian2D(num_rad,
                                     num_ang,
                                     Element::TRIANGLE,
                                     true /* gen. edges */,
                                     20.0,
                                     M_PI * 2.0,
                                     true);
   // strategy:
   // 1) generate a fes for Lagrange elements of desired degree
   // 2) create a Grid Function using a VectorFunctionCoefficient
   // 4) use mesh_ptr->NewNodes(nodes, true) to set the mesh nodes

   // Problem: fes does not own fec, which is generated in this function's scope
   // Solution: the grid function can own both the fec and fes
   H1_FECollection *fec = new H1_FECollection(degree, 2 /* = dim */);
   FiniteElementSpace *fes =
       new FiniteElementSpace(&mesh, fec, 2, Ordering::byVDIM);
   // This lambda function transforms from (r,\theta) space to (x,y) space
   auto xy_fun = [](const Vector &rt, Vector &xy)
   {
      double r_far = 20.0;
      double r = rt(0);
      double theta = rt(1);
      double ratio = 12.0;
      double delta = 2.0; // We will have to experiment with this
      double rf = 1.0 + tanh(delta * (rt(0) / r_far - 1.0)) / tanh(delta);
      double a = sqrt((1 + ratio) / (ratio - 1));
      xy(0) = a * (rf * r_far + 1) * cos(theta); // need +a to shift r away from origin
      xy(1) = a * (rf * r_far + 1) * sin(theta);
   };
   VectorFunctionCoefficient xy_coeff(2, xy_fun);
   GridFunction *xy = new GridFunction(fes);
   xy->MakeOwner(fec);
   xy->ProjectCoefficient(xy_coeff);

   mesh.NewNodes(*xy, true);
   return mesh;
}
