/// Solve the Navier-Stokes MMS verification problem
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
      "viscous": true,
      "mu": 1.0,
      "Re": 10.0,
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
/// \param[in] degree - polynomial degree of the mapping
/// \param[in] num_x - number of nodes in the x direction
/// \param[in] num_y - number of nodes in the y direction
/// \param[in] num_z - number of nodes in the z direction
unique_ptr<Mesh> buildCurvilinearMesh(int degree, int num_x, int num_y, int num_z);

/// \brief Defines the exact solution for the manufactured solution
/// \param[in] x - coordinate of the point at which the state is needed
/// \param[out] u - state variables stored as a 4-vector
void uexact(const Vector &x, Vector& u);

TEST_CASE( "Navier-Stokes MMS regression test 3D", "[NS-MMS]")
{
   int nx = 1;

   MPI_Init(NULL, NULL);
   
   int world_size;
   MPI_Comm_size(MPI_COMM_WORLD, &world_size);

   int world_rank;
   MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

   // const double target_error[4] = {
   //     0.0211824987, // p = 1
   //     0.0125304167, // p = 2
   //     0.0065000748, // p = 3
   //     0.0056838297  // p = 4
   // };
   // const double target_drag_error[4] = {
   //     0.0068506415, // p = 1
   //    -0.0140257032, // p = 2
   //     0.0086307715, // p = 3
   //    -0.0014324757  // p = 4
   // };

   // Just test p=1 and p=2 to keep test time limited
   for (int p = 0; p <= 1; ++p)
   {
      DYNAMIC_SECTION("...for solution degree p = " << p)
      {
         options["space-dis"]["degree"] = p;
         // std::cout << setw(3) << options << std::endl;
         
         // construct the mesh
         auto mesh = buildCurvilinearMesh(p+1, nx, nx, nx);
         //std::cout << "Number of elements " << mesh->GetNE() << '\n';

         // Create solver and set initial guess to exact
         FlowSolver<3,false> solver(MPI_COMM_WORLD, options, std::move(mesh));
         mfem::Vector state_tv(solver.getStateSize());
         solver.setState(uexact, state_tv);

         // get the initial entropy 
         solver.createOutput("entropy", options["outputs"].at("entropy"));
         MachInputs inputs({{"state", state_tv}});
         double entropy0 = solver.calcOutput("entropy", inputs);
         cout << "before time stepping, entropy is " << entropy0 << endl;

         // Solve for the state and compute error
         inputs = MachInputs({});
         solver.solveForState(inputs, state_tv);
         auto &state = solver.getState();
         state.distributeSharedDofs(state_tv);
         double l2_error = solver.calcConservativeVarsL2Error(uexact, 0);
         std::cout << "l2 error = " << l2_error << std::endl;
         // REQUIRE(l2_error == Approx(target_error[p - 1]).margin(1e-10));

         // get the drag error and compare against target error
         // Note: the "exact" value of drag is set to 1.6, but this has
         // not been verified (unlike the L2 error).
         // auto drag_opts = R"({ 
         //    "boundaries": [1, 3]
         // })"_json;
         // solver.createOutput("drag", drag_opts);
         // inputs = MachInputs({
         //    {"state", state.gridFunc()}
         // });
         // double drag_error = solver.calcOutput("drag", inputs) - 1.6;
         // REQUIRE(drag_error == Approx(target_drag_error[p-1]).margin(1e-10));
      }
   }
   MPI_Finalize();
}

TEST_CASE( "Navier-Stokes MMS exact solution 3D", "[NS-MMS-exact]")
{  
   int dim = 3;
   int num_state = dim + 2;
   Mesh mesh = Mesh::MakeCartesian3D(1, 1, 1, 
                                     Element::TETRAHEDRON, 1.0, 1.0, 1.0, Ordering::byVDIM);
   for (int p = 0; p <=1 ; ++p)
   {
      std::unique_ptr<FiniteElementCollection> fec(new SBPCollection(p, dim));
      std::unique_ptr<FiniteElementSpace> fes(new FiniteElementSpace(&mesh, fec.get(), dim, Ordering::byVDIM));
      GridFunction nodes(fes.get());
      mesh.GetNodes(nodes); 
      const int nNodes = nodes.Size() / dim;
      std::cout << "num nodes: " << nodes.Size() << std::endl;
      mfem::Vector x; x.SetSize(3); 
      mfem::Vector u; u.SetSize(num_state);
      for (int i = 0; i < nNodes; ++i)
      {
         for (int j = 0; j < dim; ++j)
         {
            x(j) = nodes(j * nNodes + i);
         }
         uexact(x,u);
         std::cout << u(0) << " " << u(1) << " " << u(2) << " " << u(3) << " " << u(4) << "\n";
      }
   }
}

unique_ptr<Mesh> buildCurvilinearMesh(int degree, int num_x, int num_y, int num_z)
{
   Mesh mesh = Mesh::MakeCartesian3D(num_x, num_y, num_z, 
                                     Element::TETRAHEDRON, 1.0, 1.0, 1.0, Ordering::byVDIM);
   
   for (int i = 0; i < mesh.GetNBE(); ++i)
   {
      std::cout << mesh.GetBdrAttribute(i) << "\n";
   }
   // // strategy:
   // // 1) generate a fes for Lagrange elements of desired degree
   // // 2) create a Grid Function using a VectorFunctionCoefficient
   // // 4) use mesh_ptr->NewNodes(nodes, true) to set the mesh nodes
   // H1_FECollection *fec = new H1_FECollection(degree, 3 /* = dim */);
   // FiniteElementSpace *fes = new FiniteElementSpace(&mesh, fec, 3,
   //                                                  Ordering::byVDIM);

   // auto xy_fun = [](const Vector& xi, Vector &x)
   // {
   //    x(0) = xi(0) + (1.0/40.0)*sin(2.0*M_PI*xi(0))*sin(2.0*M_PI*xi(1));
   //    x(1) = xi(1) + (1.0/40.0)*sin(2.0*M_PI*xi(1))*sin(2.0*M_PI*xi(0));
   // };
   // VectorFunctionCoefficient xy_coeff(3, xy_fun);
   // GridFunction *xy = new GridFunction(fes);
   // xy->MakeOwner(fec);
   // xy->ProjectCoefficient(xy_coeff);
   // mesh.NewNodes(*xy, true);
   return make_unique<Mesh>(mesh);
}

void uexact(const Vector &x, Vector& q)
{
   const double r_0 = 1.0;
   const double r_xyz = 1.0;
   const double u_0 = 0.0;
   const double v_0 = 0.0;
   const double w_0 = 0.0;
   const double T_0 = 1.0;
            
   q(0) = r_0 + r_0*0.1*sin(2*r_xyz*M_PI*x(0))*sin(2*r_xyz*M_PI*x(1))*sin(2*r_xyz*M_PI*x(2));
   q(1) = u_0*((pow(x(0),3)/3. - pow(x(0),2)/2.) + (pow(x(1),3)/3. - pow(x(1),2)/2.) + (pow(x(2),3)/3. - pow(x(2),2)/2.)); 
   q(2) = v_0*((pow(x(0),3)/3. - pow(x(0),2)/2.) + (pow(x(1),3)/3. - pow(x(1),2)/2.) + (pow(x(2),3)/3. - pow(x(2),2)/2.)); 
   q(3) = w_0*((pow(x(0),3)/3. - pow(x(0),2)/2.) + (pow(x(1),3)/3. - pow(x(1),2)/2.) + (pow(x(2),3)/3. - pow(x(2),2)/2.)); 
   double T = T_0;
   double p = q(0) * T;
   q(4) = p/euler::gami + 0.5 * q(0) * (q(1)*q(1) + q(2)*q(2) + q(3)*q(3)); 
}