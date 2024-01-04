/// Solve the Inviscid MMS verification problem
#include <fstream>
#include <iostream>
#include <mpi.h>

#include "adept.h"
#include "catch.hpp"
#include "mfem.hpp"

//#include "navier_stokes.hpp"
#include "euler_fluxes.hpp"
#include "euler_integ.hpp"
#include "flow_solver.hpp"

using namespace std;
using namespace mfem;
using namespace mach;

// Provide the options explicitly for regression tests
auto options = R"(
{
   "print-options": false,
   "flow-param": {
      "viscous": false,
      "mu": 1.0,
      "Re": 1000000.0,
      "Pr": 0.75,
      "inviscid-mms": true
   },
   "space-dis": {
      "degree": 0,
      "lps-coeff": 1.0,
      "flux-fun": "euler",
      "basis-type": "csbp"
   },
   "time-dis": {
      "steady": true,
      "steady-abstol": 1e-12,
      "steady-restol": 1e-10,
      "type": "PTC",
      "dt": 1,
      "cfl": 1.0,
      "res-exp": 2.5
   },
   "nonlin-solver": {
      "printlevel": 1,
      "maxiter": 50,
      "reltol": 1e-1,
      "abstol": 1e-12
   },
   "lin-solver": {
      "type": "hyprefgmres",
      "printlevel": 0,
      "filllevel": 3,
      "maxiter": 100,
      "reltol": 1e-2,
      "abstol": 1e-12
   },
   "lin-prec": {
      "type": "hypreilu",
      "lev-fill": 4
   },
   "bcs": {
      "far-field": [1, 2, 3, 4, 5, 6]
   },
   "outputs":
   {
      "drag": {
         "boundaries": [2]
      },
      "entropy": {}
   }
})"_json;

/// Generate smoothly perturbed mesh 
/// \param[in] num_x - number of nodes in the x direction
/// \param[in] num_y - number of nodes in the y direction
/// \param[in] num_z - number of nodes in the z direction
unique_ptr<Mesh> buildCurvilinearMesh(int num_x, int num_y, int num_z);

/// \brief Defines the exact solution for the manufactured solution
/// \param[in] x - coordinate of the point at which the state is needed
/// \param[out] u - state variables stored as a 4-vector
void uexact(const Vector &x, Vector& u);

TEST_CASE( "Inviscid 3D MMS inital norm convergance test", "[Inviscid-MMS]")
{
   int nx;
   // Just test p=1 and p=2 to keep test time limited
   for (int p = 0; p <= 1; ++p)
   {  nx = 5;
      double res_norm[3];
      for (int r = 0; r <= 2; ++r)
      {
         DYNAMIC_SECTION("...for solution degree p = " << p << "... for refinement level = " << r+1)
         {
            options["space-dis"]["degree"] = p;
            nx *= (r+1);
            // construct the mesh
            auto mesh = buildCurvilinearMesh(nx, nx, nx);
            std::cout << "Number of elements " << mesh->GetNE() << '\n';

            // Create solver and set initial guess to exact
            FlowSolver<3,false> solver(MPI_COMM_WORLD, options, std::move(mesh));
            mfem::Vector state_tv(solver.getStateSize());
            std::cout << "Number of nodes = " << solver.getStateSize() << "\n";
            solver.setState(uexact, state_tv);

            // set Mach inputs and compute the initial residual norm
            MachInputs inputs({{"state", state_tv}});
            res_norm[r]  = solver.calcResidualNorm(inputs);
            std::cout << "Overall initial residual norm is: " << res_norm[r] << "\n";

         }         
      }
   }

}

TEST_CASE("Inviscid 3D MMS solve", "[Inviscid-MMS]")
{
    int nx = 5;
    
    // Build mesh and print num of elements
    auto mesh = buildCurvilinearMesh(nx,nx,nx);
    std::cout << "Number of elements " << mesh->GetNE() << '\n';

    // Create solver and set initial guess to exact
    FlowSolver<3,false> solver(MPI_COMM_WORLD, options, std::move(mesh));
    mfem::Vector state_var(solver.getStateSize());
    solver.setState(uexact, state_var);

    // set Mach inputs and compute initial residual norm and print it

}

unique_ptr<Mesh> buildCurvilinearMesh(int num_x, int num_y, int num_z)
{
   Mesh mesh = Mesh::MakeCartesian3D(num_x, num_y, num_z, 
                                     Element::TETRAHEDRON, 1.0, 1.0, 1.0, Ordering::byVDIM);
   // just send out a unique pointer of the cartesian mesh generated here                               
   return make_unique<Mesh>(mesh);
}

void uexact(const Vector &x, Vector& q)
{
   const double rhop  = 0.1;
   const double rho0  = 1.0;
   const double up    = 0.1;
   const double u0    = 0.1;
   const double trans = 0.0;
   const double scale = 1.0;
   const double T0    = 1.0;
   const double Tp    = 0.1;

   q(0) = rho0 + rhop*(sin(M_PI*pow((x(0) + trans)/scale,2.0)))*sin(M_PI*(x(1)+trans)/scale);
   q(1) = 4*u0*((x(1)+trans)/scale)*(1 - ((x(1) +trans)/scale)) + 
          up*sin(2*M_PI*(x(1)+trans)/scale)*(sin(M_PI*pow((x(0)+trans)/scale,2.0)));
   q(2) = -up*(sin(2*M_PI*pow((x(0)+trans)/scale,2.0)))*sin(M_PI*(x(1)+trans)/scale);
   q(3) = 0.0;
   double Tem  = T0 + Tp*(pow(((x(0)+trans)/scale),4.0) - 2.0*pow(((x(0)+trans)/scale),3.0) 
                 + pow(((x(0)+trans)/scale),2.0) + pow(((x(1)+trans)/scale),4.0) 
                 - 2.0*pow(((x(1)+trans)/scale),3.0) + pow(((x(1)+trans)/scale),2.0));
   double p    = q(0)*Tem;
   q(4) = (p/euler::gami) + q(0)*(q(1)*q(1) + q(2)*q(2) + q(3)*q(3))*0.5;
   q(1) *= q(0);
   q(2) *= q(0);
   q(3) *= q(0);
}