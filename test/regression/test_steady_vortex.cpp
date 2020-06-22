/// Solve the steady isentropic vortex problem on a quarter annulus
#include "adept.h"

#include "catch.hpp"
#include "mfem.hpp"
#include "euler.hpp"
#include <fstream>
#include <iostream>

using namespace std;
using namespace mfem;
using namespace mach;

/// NOTE: Not sure how I'll iterate over order, etc.
bool entvarg;

/// \brief Defines the exact solution for the steady isentropic vortex
/// \param[in] x - coordinate of the point at which the state is needed
/// \param[out] u - conservative variables stored as a 4-vector
void uexact(const Vector &x, Vector& q);

/// Generate quarter annulus mesh 
/// \param[in] degree - polynomial degree of the mapping
/// \param[in] num_rad - number of nodes in the radial direction
/// \param[in] num_ang - number of nodes in the angular direction
std::unique_ptr<Mesh> buildQuarterAnnulusMesh(int degree, int num_rad,
                                              int num_ang);

TEMPLATE_TEST_CASE_SIG("Steady Vortex Solver Regression Test",
                       "[euler]", ((bool entvar), entvar), false, true)
{
   entvarg = entvar;   

   const char *options_file = "test_steady_vortex_options.json";
  
    // Parse command-line options
    int argc; char ** argv;
    OptionsParser args(argc, argv);
    int degree = 2.0;
    int nx = 1;
    int ny = 1;
    args.AddOption(&options_file, "-o", "--options",
                  "Options file to use.");
    args.AddOption(&degree, "-d", "--degree", "poly. degree of mesh mapping");
    args.AddOption(&nx, "-nr", "--num-rad", "number of radial segments");
    args.AddOption(&ny, "-nt", "--num-thetat", "number of angular segments");
    args.Parse();
    if (!args.Good())
    {
       args.PrintUsage(cout);
    }
  
   for (int h = 1; h <= 4; ++h)
   {
      DYNAMIC_SECTION("...for mesh sizing h = " << h)
      {
         nx = h; ny = h;
          // construct the solver, set the initial condition, and solve
          string opt_file_name(options_file);
          unique_ptr<Mesh> smesh = buildQuarterAnnulusMesh(degree, nx, ny);
          std::cout <<"Number of elements " << smesh->GetNE() <<'\n';

         auto solver = createSolver<EulerSolver<2, entvar>>(opt_file_name,
                                                               move(smesh));    //unique_ptr<AbstractSolver> solver(new EulerSolver<2>(opt_file_name, nullptr));
          solver->initDerived();

          solver->setInitialCondition(uexact);

          double l_error = solver->calcL2Error(uexact, 0);
          double res_error = solver->calcResidualNorm();

          solver->solveForState();
          l_error = solver->calcL2Error(uexact, 0);
          //res_error = solver->calcResidualNorm();
          //double drag = abs(solver->calcOutput("drag") - (-1 / mach::euler::gamma));

         double target;
         switch(h)
         {
            case 1: 
               target = !entvarg ? 0.0700148195 :  0.0690131081;
               break;
            case 2: 
               target = !entvarg ? 0.0260625842 :  0.0224304871;
               break;
            case 3: 
               target = !entvarg ? 0.0129909277 :  0.0107753424;
               break;
            case 4: 
               target = !entvarg ? 0.0079317615 :  0.0064387612;
               break;
         }
         REQUIRE(l_error == Approx(target).margin(1e-10));
      }
   }
#ifdef MFEM_USE_PETSC
   MFEMFinalizePetsc();
#endif
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
   u(1) = rho*a*Ma*sin(theta);
   u(2) = -rho*a*Ma*cos(theta);
   u(3) = press/euler::gami + 0.5*rho*a*a*Ma*Ma;

   if (entvarg == false)
   {
      q = u;
   }
   else
   {
      calcEntropyVars<double, 2>(u.GetData(), q.GetData());
   }
}

unique_ptr<Mesh> buildQuarterAnnulusMesh(int degree, int num_rad, int num_ang)
{
   auto mesh_ptr = unique_ptr<Mesh>(new Mesh(num_rad, num_ang,
                                             Element::TRIANGLE, true /* gen. edges */,
                                             2.0, M_PI*0.5, true));
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
      xy(0) = (rt(0) + 1.0)*cos(rt(1)); // need + 1.0 to shift r away from origin
      xy(1) = (rt(0) + 1.0)*sin(rt(1));
   };
   VectorFunctionCoefficient xy_coeff(2, xy_fun);
   GridFunction *xy = new GridFunction(fes);
   xy->MakeOwner(fec);
   xy->ProjectCoefficient(xy_coeff);

   mesh_ptr->NewNodes(*xy, true);
   return mesh_ptr;
}
