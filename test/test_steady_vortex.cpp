#include "catch.hpp"
#include "mfem.hpp"
#include "euler.hpp"
#include "euler_fluxes.hpp"
#include <fstream>
#include <iostream>

/// \brief Defines the exact solution for the steady isentropic vortex
/// \param[in] x - coordinate of the point at which the state is needed
/// \param[out] u - conservative variables stored as a 4-vector
void uexact(const mfem::Vector &x, mfem::Vector& u);

/// Generate quarter annulus mesh 
/// \param[in] degree - polynomial degree of the mapping
/// \param[in] num_rad - number of nodes in the radial direction
/// \param[in] num_ang - number of nodes in the angular direction
std::unique_ptr<mfem::Mesh> buildQuarterAnnulusMesh(int degree, int num_rad,
                                             int num_ang);


TEST_CASE( "Steady Vortex Problem", "[stdy-vrtx]")
{  
   using namespace std;
   using namespace mfem;
   using namespace mach;

   {
   #ifdef MFEM_USE_MPI
      // Initialize MPI if parallel
      MPI_Init(NULL, NULL);
   #endif
      const char *options_file = "test_steady_vortex_options.json";
      int degree = 2.0;
      int nx = 1.0;
      int ny = 1.0;


      
         // construct the solver, set the initial condition, and solve
         string opt_file_name(options_file);
         const int dim = 2;
         std::unique_ptr<Mesh> smesh = buildQuarterAnnulusMesh(degree, nx, ny);
         std::cout <<"Number of elements " << smesh->GetNE() <<'\n';
         ofstream sol_ofs("steady_vortex_mesh.vtk");
         sol_ofs.precision(14);
         smesh->PrintVTK(sol_ofs,3);
         EulerSolver solver(opt_file_name, move(smesh), dim);
         solver.setInitialCondition(uexact);
         solver.printSolution("init", degree+1);
         mfem::out << "\n|| rho_h - rho ||_{L^2} = " 
                  << solver.calcL2Error(uexact, 0) << '\n' << endl;
         mfem::out << "\ninitial residual norm = " << solver.calcResidualNorm()
                  << endl;
         solver.solveForState();
         mfem::out << "\nfinal residual norm = " << solver.calcResidualNorm()
                  << endl;
         mfem::out << "\n|| rho_h - rho ||_{L^2} = " 
                  << solver.calcL2Error(uexact, 0) << '\n' << endl;

         double error = solver.calcL2Error(uexact, 0);


   #ifdef MFEM_USE_MPI
      MPI_Finalize();
   #endif

   REQUIRE( error == Approx(0.0697252));
   }
}

// Exact solution; note that I reversed the flow direction to be clockwise, so
// the problem and mesh are consistent with the LPS paper (that is, because the
// triangles are subdivided from the quads using the opposite diagonal)
void uexact(const mfem::Vector &x, mfem::Vector& u)
{
   u.SetSize(4);
   double ri = 1.0;
   double Mai = 0.5; //0.95 
   double rhoi = 2.0;
   double prsi = 1.0/mach::euler::gamma;
   double rinv = ri/sqrt(x(0)*x(0) + x(1)*x(1));
   double rho = rhoi*pow(1.0 + 0.5*mach::euler::gami*Mai*Mai*(1.0 - rinv*rinv),
                         1.0/mach::euler::gami);
   double Ma = sqrt((2.0/mach::euler::gami)*( ( pow(rhoi/rho, mach::euler::gami) ) * 
                    (1.0 + 0.5*mach::euler::gami*Mai*Mai) - 1.0 ) );
   double theta;
   if (x(0) > 1e-15)
   {
      theta = atan(x(1)/x(0));
   }
   else
   {
      theta = M_PI/2.0;
   }
   double press = prsi* pow( (1.0 + 0.5*mach::euler::gami*Mai*Mai) / 
                 (1.0 + 0.5*mach::euler::gami*Ma*Ma), mach::euler::gamma/mach::euler::gami);
   double a = sqrt(mach::euler::gamma*press/rho);

   u(0) = rho;
   u(1) = rho*a*Ma*sin(theta);
   u(2) = -rho*a*Ma*cos(theta);
   u(3) = press/mach::euler::gami + 0.5*rho*a*a*Ma*Ma;
}

std::unique_ptr<mfem::Mesh> buildQuarterAnnulusMesh(int degree, int num_rad, int num_ang)
{
   auto mesh_ptr = std::unique_ptr<mfem::Mesh>(new mfem::Mesh(num_rad, num_ang,
                                             mfem::Element::TRIANGLE, true /* gen. edges */,
                                             2.0, M_PI*0.5, true));
   // strategy:
   // 1) generate a fes for Lagrange elements of desired degree
   // 2) create a Grid Function using a VectorFunctionCoefficient
   // 4) use mesh_ptr->NewNodes(nodes, true) to set the mesh nodes
   
   // Problem: fes does not own fec, which is generated in this function's scope
   // Solution: the grid function can own both the fec and fes
   mfem::H1_FECollection *fec = new mfem::H1_FECollection(degree, 2 /* = dim */);
   mfem::FiniteElementSpace *fes = new mfem::FiniteElementSpace(mesh_ptr.get(), fec, 2,
                                                    mfem::Ordering::byVDIM);

   // This lambda function transforms from (r,\theta) space to (x,y) space
   auto xy_fun = [](const mfem::Vector& rt, mfem::Vector &xy)
   {
      xy(0) = (rt(0) + 1.0)*cos(rt(1)); // need + 1.0 to shift r away from origin
      xy(1) = (rt(0) + 1.0)*sin(rt(1));
   };
   mfem::VectorFunctionCoefficient xy_coeff(2, xy_fun);
   mfem::GridFunction *xy = new mfem::GridFunction(fes);
   xy->MakeOwner(fec);
   xy->ProjectCoefficient(xy_coeff);

   mesh_ptr->NewNodes(*xy, true);
   return mesh_ptr;
}
