// Solve for the steady flow around a NACA0012

// set this const expression to true in order to use entropy variables for state
constexpr bool entvar = false;

#include <random>
#include <fstream>
#include <iostream>

#include "mfem.hpp"

#include "euler_dg_cut.hpp"

using namespace std;
using namespace mfem;
using namespace mach;

std::default_random_engine gen(std::random_device{}());
std::uniform_real_distribution<double> normal_rand(-1.0, 1.0);

/// \brief Defines the random function for the jabocian check
/// \param[in] x - coordinate of the point at which the state is needed
/// \param[out] u - conservative variables stored as a 4-vector
void pert(const Vector &x, Vector &p);

/// \brief Defines the exact solution for the steady isentropic vortex
/// \param[in] x - coordinate of the point at which the state is needed
/// \param[out] u - state variables stored as a 4-vector
void uexact(const Vector &x, Vector &u);
/// Generate quarter annulus mesh
/// \param[in] N - number of elements in x-y direction
unique_ptr<Mesh>  buildMesh(int degree, int num_rad, int num_ang);
int main(int argc, char *argv[])
{
   const char *options_file = "ellipse_potential_flow_options.json";
   // Initialize MPI
   int num_procs, rank;
   MPI_Init(&argc, &argv);
   MPI_Comm_size(MPI_COMM_WORLD, &num_procs);
   MPI_Comm_rank(MPI_COMM_WORLD, &rank);
   ostream *out = getOutStream(rank);
   int Nx = 8;
   int Ny = 20;
   int degree = 1;
   // Parse command-line options
   OptionsParser args(argc, argv);
   args.AddOption(&options_file, "-o", "--options", "Options file to use.");
   args.AddOption(&Nx, "-nx", "--#elements in x-dir", "number of elements");
   args.AddOption(&Ny, "-ny", "--#elements in y-dir", "number of elements");
   args.AddOption(&degree, "-d", "--degree", "degree of mesh elements");
   args.Parse();
   if (!args.Good())
   {
      args.PrintUsage(cout);
      return 1;
   }

   try
   {
      // construct the solver, set the initial condition, and solve
      unique_ptr<Mesh> smesh = buildMesh(degree, Nx, Ny);
      *out << "Number of elements " << smesh->GetNE() << '\n';
      ofstream sol_ofs("ellipse_mesh_dg_cut_init.vtk");
      sol_ofs.precision(14);
      smesh->PrintVTK(sol_ofs);
   }

   catch (MachException &exception)
   {
      exception.print_message();
   }
   catch (std::exception &exception)
   {
      cerr << exception.what() << endl;
   }

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
#if 1
/// use this for flow over an ellipse
void uexact(const Vector &x, Vector &q)
{
   q.SetSize(4);
   Vector u(4);
   double theta;
   double Ma = 0.2;
   double rho = 1.0;
   double p = 1.0 / euler::gamma;
   /// ellipse parameters
   double xc = 10.0;
   double yc = 10.0;
   double a = 2.5;
   double b = sqrt(a * (a - 1));
   double s =
       ((x(0) - xc) * (x(0) - xc)) - ((x(1) - yc) * (x(1) - yc)) - 4.0 * b * b;
   double t = 2.0 * (x(0) - xc) * (x(1) - yc);
   theta = atan2(t, s);
   double signx = 1.0;
   if (x(0) - xc < 0)
   {
      signx = -1.0;
   }
   double r = sqrt(t * t + s * s);
   double xi = 0.5 * (x(0) - xc + (signx * sqrt(r) * cos(theta / 2.0)));
   double eta = 0.5 * (x(1) - yc + (signx * sqrt(r) * sin(theta / 2.0)));
   double term_a = xi * xi - eta * eta - a * a;
   double term_b = xi * xi - eta * eta - b * b;
   double term_c = 4.0 * xi * xi * eta * eta;
   double term_d = (term_b * term_b) + term_c;
   u(0) = rho;
   u(1) = rho * Ma * ((term_a * term_b) + term_c) / term_d;
   u(2) = -rho * Ma * 2.0 * xi * eta * (term_b - term_a) / term_d;
   u(3) = p / euler::gami + 0.5 * Ma * Ma;
   if (entvar == false)
   {
      q = u;
   }
   else
   {
      calcEntropyVars<double, 2>(u.GetData(), q.GetData());
   }
}
#endif
#if 0
unique_ptr<Mesh>  buildMesh(int degree)
{
   int ref_levels = 6;
   // auto mesh_ptr = unique_ptr<Mesh>(new Mesh(num_rad, num_ang,
   //                                          Element::QUADRILATERAL, true /* gen. edges */,
   //                                          40.0, 2*M_PI, true));
   const char *mesh_file = "periodic_rectangle.mesh";
   auto mesh_ptr = unique_ptr<Mesh>(new Mesh(mesh_file, 1, 1));

   for (int l = 0; l < ref_levels; l++)
   {
      mesh_ptr->UniformRefinement();
   }
   cout << "Number of elements " << mesh_ptr->GetNE() << '\n';
   // strategy:
   // 1) generate a fes for Lagrange elements of desired degree
   // 2) create a Grid Function using a VectorFunctionCoefficient
   // 4) use mesh_ptr->NewNodes(nodes, true) to set the mesh nodes

   // Problem: fes does not own fec, which is generated in this function's scope
   // Solution: the grid function can own both the fec and fes
   H1_FECollection *fec = new H1_FECollection(degree, 2 /* = dim */);
   FiniteElementSpace *fes = new FiniteElementSpace(mesh_ptr.get(), fec, 2,
                                                    Ordering::byVDIM);

   // This lambda function transforms from (r,\theta) space to (x,y) space
   auto xy_fun = [](const Vector &rt, Vector &xy) {
      double r_far = 40.0;
      double a0 = 0.5;
      double b0 = a0 / 10.0;
      double delta = 3.00; // We will have to experiment with this
      double r = 1.0 + tanh(delta * (rt(0) / r_far - 1.0)) / tanh(delta);
      double theta = rt(1);
      double b = b0 + (a0 - b0) * r;
      xy(0) = a0 * (r * r_far + 1.0) * cos(theta) + 20.0;
      xy(1) = b * (r * r_far + 1.0) * sin(theta) + 20.0;
   };

   VectorFunctionCoefficient xy_coeff(2, xy_fun);
   GridFunction *xy = new GridFunction(fes);
   xy->MakeOwner(fec);
   xy->ProjectCoefficient(xy_coeff);

   mesh_ptr->NewNodes(*xy, true);
   return mesh_ptr;
}
#endif
/// use this for circle
unique_ptr<Mesh> buildMesh(int degree, int num_rad, int num_ang)
{
   auto mesh_ptr = unique_ptr<Mesh>(new Mesh(num_rad, num_ang,
                                             Element::QUADRILATERAL, true /* gen. edges */,
                                             6.0, 2*M_PI, true));
   // strategy:
   // 1) generate a fes for Lagrange elements of desired degree
   // 2) create a Grid Function using a VectorFunctionCoefficient
   // 4) use mesh_ptr->NewNodes(nodes, true) to set the mesh nodes
   
   // Problem: fes does not own fec, which is generated in this function's scope
   // Solution: the grid function can own both the fec and fes
   H1_FECollection *fec = new H1_FECollection(degree, 2 /* = dim */);
   FiniteElementSpace *fes = new FiniteElementSpace(mesh_ptr.get(), fec, 2,
                                                    Ordering::byVDIM);

   // This lambda function transforms from (r,\theta) space to (x,y) space
   auto xy_fun = [](const Vector& rt, Vector &xy)
   {
      xy(0) = ((rt(0)+ 4.0)*cos(rt(1))) + 10.0; // need + 1.0 to shift r away from origin
      // xy(1) = (rt(0)+ 4.0)*sin(rt(1)) + 10.0 ;
      xy(1) = (rt(0)+ 1.0)*sin(rt(1)) + 10.0 ;
   };
   VectorFunctionCoefficient xy_coeff(2, xy_fun);
   GridFunction *xy = new GridFunction(fes);
   xy->MakeOwner(fec);
   xy->ProjectCoefficient(xy_coeff);

   mesh_ptr->NewNodes(*xy, true);
   return mesh_ptr;
}
