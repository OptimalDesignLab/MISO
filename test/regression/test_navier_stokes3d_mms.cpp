/// Solve the Navier-Stokes MMS verification problem
#include <fstream>
#include <iostream>
#include <mpi.h>

#include "adept.h"
#include "catch.hpp"
#include "mfem.hpp"

#include "navier_stokes.hpp"
// #include "euler_fluxes.hpp"
// #include "euler_integ.hpp"
#include "flow_solver.hpp"

using namespace std;
using namespace mfem;
using namespace miso;

// Provide the options explicitly for regression tests
auto options = R"(
{
   "print-options": false,
   "flow-param": {
      "viscous": true,
      "mu": 1.0,
      "Re": 1000000.0,
      "Pr": 0.75,
      "viscous-mms": true
   },
   "space-dis": {
      "degree": 0,
      "lps-coeff": 1.0,
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
      "no-slip-adiabatic": [1, 2, 3, 4, 5, 6]
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

TEST_CASE( "Navier-Stokes 3D MMS inital norm convergance test", "[NS-MMS]")
{
   int nx;
   // Just test p=1 and p=2 to keep test time limited
   for (int p = 0; p <= 1; ++p)
   {  nx = 10;
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

            // set Miso inputs and compute the initial residual norm
            MisoInputs inputs({{"state", state_tv}});
            res_norm[r]  = solver.calcResidualNorm(inputs);
            std::cout << "Overall initial residual norm is: " << res_norm[r] << "\n";

         }         
      }
   }

}

// TEST_CASE( "Navier-Stokes MMS exact solution 3D", "[NS-MMS-exact]")
// {  
//    int dim = 3;
//    int num_state = dim + 2;
//    Mesh mesh = Mesh::MakeCartesian3D(1, 1, 1, 
//                                      Element::TETRAHEDRON, 1.0, 1.0, 1.0, Ordering::byVDIM);
//    for (int p = 0; p <=1 ; ++p)
//    {
//       std::unique_ptr<FiniteElementCollection> fec(new SBPCollection(p, dim));
//       std::unique_ptr<FiniteElementSpace> fes(new FiniteElementSpace(&mesh, fec.get(), dim, Ordering::byVDIM));
//       GridFunction nodes(fes.get());
//       mesh.GetNodes(nodes); 
//       const int nNodes = nodes.Size() / dim;
//       std::cout << "num nodes: " << nodes.Size() << std::endl;
//       mfem::Vector x; x.SetSize(3); 
//       mfem::Vector u; u.SetSize(num_state);
//       for (int i = 0; i < nNodes; ++i)
//       {
//          for (int j = 0; j < dim; ++j)
//          {
//             x(j) = nodes(j * nNodes + i);
//          }
//          uexact(x,u);
//          std::cout << u(0) << " " << u(1) << " " << u(2) << " " << u(3) << " " << u(4) << "\n";
//       }
//    }
// }

unique_ptr<Mesh> buildCurvilinearMesh(int num_x, int num_y, int num_z)
{
   Mesh mesh = Mesh::MakeCartesian3D(num_x, num_y, num_z, 
                                     Element::TETRAHEDRON, 1.0, 1.0, 1.0, Ordering::byVDIM);
   // just send out a unique pointer of the cartesian mesh generated here                               
   return make_unique<Mesh>(mesh);
}

void uexact(const Vector &x, Vector& q)
{
   const double r_0 = 1.0;
   const double r_xyz = 1.0;
   const double u_0 = 0.5;
   const double v_0 = 0.5;
   const double w_0 = 0.5;
   const double T_0 = 1.0;

   q[0] = r_0 + r_0*0.1*sin(2*r_xyz*M_PI*x[0])*sin(2*r_xyz*M_PI*x[1])*sin(2*r_xyz*M_PI*x[3]);
   q[1] = u_0*((pow(x[0],3)/3. - pow(x[0],2)/2.) + (pow(x[1],3)/3. - pow(x[1],2)/2.) + (pow(x[2],3)/3. - pow(x[2],2)/2.)); 
   q[2] = v_0*((pow(x[0],3)/3. - pow(x[0],2)/2.) + (pow(x[1],3)/3. - pow(x[1],2)/2.) + (pow(x[2],3)/3. - pow(x[2],2)/2.)); 
   q[3] = w_0*((pow(x[0],3)/3. - pow(x[0],2)/2.) + (pow(x[1],3)/3. - pow(x[1],2)/2.) + (pow(x[2],3)/3. - pow(x[2],2)/2.)); 
   double T = T_0;
   double p = q[0] * T;
   q[4] = p/euler::gami + 0.5 * q[0] * (q[1]*q[1] + q[2]*q[2] + q[3]*q[3]); 
   q[1] *= q[0];
   q[2] *= q[0];
   q[3] *= q[0];
}