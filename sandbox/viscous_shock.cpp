/// Solve the viscous shock verification problem

#include "mfem.hpp"
#include "navier_stokes.hpp"
#include <fstream>
#include <iostream>

using namespace std;
using namespace mfem;
using namespace mach;

/// Find root of `func` using bisection
/// \param[in] func - function to find root of 
/// \param[in] xl - left bracket of root
/// \param[in] xr - right bracket of root
/// \param[in] ftol - absolute tolerance for root function
/// \param[in] xtol - absolute tolerance for root value
/// \param[in] maxiter - maximum number of iterations
double bisection(std::function<double(double)> func, double xl, double xr,
                 double ftol, double xtol, int maxiter);

/// Defines the right-hand side of Equation (7.5) in "Entropy stable spectral
/// collocation schemes for the Navier-Stokes questions: discontinuous
/// interfaces."  See also Fisher's thesis in the appendix.
/// \param[in] Re - Reynolds number
/// \param[in] Ma - Mach number
/// \param[in] v - velocity ration u/u_L
/// \returns the right hand side of Equation (7.5)
double shockEquation(double Re, double Ma, double v);

/// \brief Defines the initial condition for the viscous shock
/// \param[in] x - coordinate of the point at which the state is needed
/// \param[out] u - conservative variables stored as a 4-vector
void u0(const mfem::Vector &x, mfem::Vector& u);

/// Generate smoothly perturbed mesh 
/// \param[in] degree - polynomial degree of the mapping
/// \param[in] num_x - number of nodes in the x direction
/// \param[in] num_y - number of nodes in the y direction
std::unique_ptr<Mesh> buildCurvilinearMesh(int degree, int num_x, int num_y);

int main(int argc, char *argv[])
{
#ifdef MFEM_USE_MPI
   // Initialize MPI if parallel
   MPI_Init(&argc, &argv);
#endif
   // Parse command-line options
   OptionsParser args(argc, argv);
   const char *options_file = "viscous_shock_options.json";
   int degree = 2.0;
   int nx = 1.0;
   int ny = 1.0;
   args.AddOption(&options_file, "-o", "--options",
                  "Options file to use.");
   args.AddOption(&degree, "-d", "--degree", "poly. degree of mesh mapping");
   args.AddOption(&nx, "-nx", "--num-x", "number of x-direction segments");
   args.AddOption(&ny, "-ny", "--num-y", "number of y-direction segments");
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
      unique_ptr<Mesh> smesh = buildCurvilinearMesh(degree, nx, ny);
      std::cout <<"Number of elements " << smesh->GetNE() <<'\n';
      ofstream sol_ofs("viscous_shock_mesh.vtk");
      sol_ofs.precision(14);
      smesh->PrintVTK(sol_ofs, 3);
      unique_ptr<AbstractSolver> solver(new NavierStokesSolver<2>(opt_file_name, move(smesh)));

      // Define the initial condition function
      solver->setInitialCondition(u0);
      solver->printSolution("init", degree+1);

      mfem::out << "\n|| rho_h - rho ||_{L^2} = " 
                << solver->calcL2Error(u0, 0) << '\n' << endl;
      mfem::out << "\ninitial residual norm = " << solver->calcResidualNorm()
                << endl;
#if 0
      solver.solveForState();
      mfem::out << "\nfinal residual norm = " << solver.calcResidualNorm()
                << endl;
      mfem::out << "\n|| rho_h - rho ||_{L^2} = " 
                << solver.calcL2Error(uexact, 0) << endl;
      mfem::out << "\nDrag error = "
                << abs(solver.calcOutput("drag") - (-1 / mach::euler::gamma)) << endl
                << endl;
#endif
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

double bisection(std::function<double(double)> func, double xl, double xr,
                 double ftol, double xtol, int maxiter)
{

   double fl = func(xl);
   double fr = func(xr);
   if (fl*fr > 0.0)
   {
      cerr << "bisection: func(xl)*func(xr) is positive." << endl;
      throw(-1);
   }
   double xm = 0.5*(xl + xr);
   double fm = func(xm);
   int iter = 0;
   while ( (abs(fm) > ftol) && (abs(xr - xl) > xtol) && (iter < maxiter) )
   {
      iter++;
      //cout << "iter = " << iter << ": f(x) = " << fm << endl;
      if (fm*fl < 0.0)
      {
         xr = xm;
         fr = fm;
      }
      else if (fm*fr < 0.0)
      {
         xl = xm;
         fl = fm;
      }
      else {
         break;
      }
      xm = 0.5*(xl + xr);
      fm = func(xm);
   }
   if (iter >= maxiter)
   {
      cerr << "bisection: failed to find a solution in maxiter." << endl;
      throw(-1);
   }
   return xm;
}

double shockEquation(double Re, double Ma, double v)
{
   double vf = (2.0 + euler::gami * Ma * Ma) / ((euler::gamma + 1) * Ma * Ma);
   double alpha = (euler::gami) / (2 * euler::gamma * Re);
   double r = (1 + vf) / (1 - vf);
   double a = abs((v - 1) * (v - vf));
   double b = (1 + vf) / (1 - vf);
   double c = abs((v - 1) / (v - vf));
   return 0.5 * alpha * (log(a) + b * log(c));
}

// Exact solution initially
void u0(const mfem::Vector &x, mfem::Vector& u)
{
   // define a lambda function for equation (7.5)
   double Re = 10.0; // !!!!! Values from options file are ignored
   double Ma = 2.5;
   double vf = (2.0 + euler::gami*Ma*Ma)/((euler::gamma + 1)*Ma*Ma);
   double v;
   double ftol = 1e-10;
   double xtol = 1e-10;
   int maxiter = 50;
   if (x(0) < -1.0/30.0)
   {
      v = 1.0;
   }
   else if (x(0) > 1.0/30.0)
   {
      v = vf;
   }
   else
   {
      auto func = [&](double vroot)
      {
         return x(0) - shockEquation(Re, Ma, vroot);
      };
      v = bisection(func, 1.00001*vf, 0.99999, ftol, xtol, maxiter);
   }

   u(1) = v*Ma;
   // now get the density and energy
   u(0) = 1/u(1);
   u(3) = 0.5*euler::gami*u(0)*u(1)*u(1)/euler::gamma + Ma*Ma;
   u(1) = 1.0; // mass flow rate is 1.0*dy 
   u(2) = 0.0; // zero vertical velocity
}

std::unique_ptr<Mesh> buildCurvilinearMesh(int degree, int num_x, int num_y)
{
   auto mesh_ptr = unique_ptr<Mesh>(new Mesh(num_x, num_y,
                                             Element::TRIANGLE, true /* gen. edges */,
                                             1.0, 1.0, true));
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
   auto xy_fun = [](const Vector& xi, Vector &x)
   {
      x(0) = xi(0) + (1.0/40.0)*sin(2.0*M_PI*xi(0))*sin(2.0*M_PI*xi(1));
      x(1) = xi(1) + (1.0/40.0)*sin(2.0*M_PI*xi(1))*sin(2.0*M_PI*xi(0));
      x(0) = x(0) - 0.5; //2.0*x(0) - 1.0;
      x(1) -= 0.5;
   };
   VectorFunctionCoefficient xy_coeff(2, xy_fun);
   GridFunction *xy = new GridFunction(fes);
   xy->MakeOwner(fec);
   xy->ProjectCoefficient(xy_coeff);

   mesh_ptr->NewNodes(*xy, true);
   return mesh_ptr;
}