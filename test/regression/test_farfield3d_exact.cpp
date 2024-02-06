/// Solve the Navier-Stokes MMS verification problem
#include <fstream>
#include <iostream>
#include <mpi.h>
#include <time.h>

#include "adept.h"
#include "catch.hpp"
#include "mfem.hpp"

#include "navier_stokes.hpp"
#include "euler_fluxes.hpp"
#include "euler_integ.hpp"
#include "flow_solver.hpp"

using namespace std;
using namespace mfem;
using namespace mach;

/// \brief Defines the exact solution for the manufactured solution + perturb by 5%
/// \param[in] x - coordinate of the point at which the state is needed
/// \param[out] u - state variables stored as a 4-vector
void uexact_pert(const Vector &x, Vector& q);

/// \brief Defines the exact solution for the manufactured solution
/// \param[in] x - coordinate of the point at which the state is needed
/// \param[out] u - state variables stored as a 4-vector
void uexact(const Vector &x, Vector& q);

// Provide the options explicitly for regression tests

auto options = R"(
{
   "print-options": false,
   "flow-param": {
      "viscous": true,
      "mach": 0.5,
      "mu": 1.0,
      "Re": 1000.0,
      "Pr": 0.75,
      "viscous-mms": false
   },
   "space-dis": {
      "degree": 1,
      "lps-coeff": 1.0,
      "basis-type": "csbp"
   },
   "time-dis": {
      "steady": true,
      "steady-abstol": 1e-12,
      "steady-restol": 1e-10,
      "type": "PTC",
      "dt": 0.1,
      "cfl": 0.5,
      "res-exp": 2.0
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
// {
//     "flow-param": {
//        "mach": 0.5, 
//        "mu": 1.0,
//        "Re": 1000.0,
//        "Pr": 0.75,
//        "viscous-mms": true
//     },
//     "space-dis": {
//        "degree": 1,
//        "lps-coeff": 1.0,
//        "basis-type": "csbp"
//     },
//     "steady": true,
//     "time-dis": {
//        "steady": true,
//        "steady-abstol": 1e-12,
//        "steady-restol": 1e-10,
//        "const-cfl": true,
//        "ode-solver": "PTC",
//        "t-final": 100,
//        "dt": 0.001,
//        "cfl": 0.1,
//        "res-exp": 1.0
//     },
//     "nonlin-solver": {
//        "printlevel": 1,
//        "maxiter": 50,
//        "reltol": 1e-1,
//        "abstol": 1e-12
//     },
//     "lin-solver": {
//        "type": "gmres",
//        "printlevel": 1,
//        "filllevel": 3,
//        "maxiter": 100,
//        "reltol": 1e-2,
//        "abstol": 1e-12
//     },
//     "bcs": {
//        "far-field": [1, 1, 1, 1, 1, 1]
//     },
//     "outputs":
//     {
//        "drag": {}
//     }
//  }
TEST_CASE("Simple Far-Field test" , "NS-nonMMS")
{
    const int dim = 3;

    int nx = 1, ny = 1, nz = 1;
    Mesh smesh(Mesh::MakeCartesian3D(nx, ny, nz, Element::TETRAHEDRON));
    auto mesh = make_unique<Mesh>(smesh);

    // create solver and set initial conditions to uexact + small perturbations
    FlowSolver<dim, false> ns_solver(MPI_COMM_WORLD, options, std::move(mesh));
   //  mfem::Vector qfar(ns_solver.getStateSize());
   //  ns_solver.getFreeStreamState(qfar);
   //  auto uInit = [&](const Vector &x, Vector &u0) { u0 = qfar; };
    mfem::Vector state_tv(ns_solver.getStateSize());
    ns_solver.setState(uexact_pert, state_tv);

    // solve for solution
    MachInputs inputs({{"state", state_tv}});
    inputs = MachInputs({});
    ns_solver.solveForState(inputs, state_tv);
    auto &state = ns_solver.getState();
    state.distributeSharedDofs(state_tv);
    double l2_error = ns_solver.calcConservativeVarsL2Error(uexact, 0);
    //auto l2_error = ns_solver.calcStateError(uInit, state_tv);
    std::cout << "l2 error = " << l2_error << std::endl;
    // NavierStokesSolver<dim, false> ns_solver(options, std::move(mesh), MPI_COMM_WORLD);
    // ns_solver.setInitialCondition(uexact_pert);
    // ns_solver.solveForState();
    // std::cout << ns_solver.calcL2Error(uexact,0) << "\n";

}

void uexact_pert(const Vector &x, Vector& q)
{
   //  const double r_0 = 1.0;
   //  const double r_xyz = 1.0;
   //  const double u_0 = 0.0;
   //  const double v_0 = 0.0;
   //  const double w_0 = 0.0;
   //  const double T_0 = 1.0;
            
   //  q(0) = r_0 + r_0*0.1*sin(2*r_xyz*M_PI*x(0))*sin(2*r_xyz*M_PI*x(1))*sin(2*r_xyz*M_PI*x(2));
   //  q(1) = u_0*((pow(x(0),3)/3. - pow(x(0),2)/2.) + (pow(x(1),3)/3. - pow(x(1),2)/2.) + (pow(x(2),3)/3. - pow(x(2),2)/2.)); 
   //  q(2) = v_0*((pow(x(0),3)/3. - pow(x(0),2)/2.) + (pow(x(1),3)/3. - pow(x(1),2)/2.) + (pow(x(2),3)/3. - pow(x(2),2)/2.)); 
   //  q(3) = w_0*((pow(x(0),3)/3. - pow(x(0),2)/2.) + (pow(x(1),3)/3. - pow(x(1),2)/2.) + (pow(x(2),3)/3. - pow(x(2),2)/2.)); 
   //  double T = T_0;
   //  double p = q(0) * T;
   //  q(4) = p/euler::gami + 0.5 * q(0) * (q(1)*q(1) + q(2)*q(2) + q(3)*q(3));

   //  srand( (unsigned)time( NULL ) );
   q(0) = 1.0;
   q(1) = 0.5;
   q(2) = 0.0;
   q(3) = 0.0;
   q(4) = 1.9107142857142858;
    // adding random 5% perturbations to the initial condition
    for (int i = 0; i < q.Size(); ++i)
    {
        float r = (float) rand()/RAND_MAX ;
        if (r > (float)(0.5))
        { q(i) += 0.05*q(i); }
        else
        { q(i) -= 0.05*q(i); }
        std::cout << "q(" << i << "): " << q(i) << "\n";
    }
}

void uexact(const Vector &x, Vector& q)
{
   //  const double r_0 = 1.0;
   //  const double r_xyz = 1.0;
   //  const double u_0 = 0.0;
   //  const double v_0 = 0.0;
   //  const double w_0 = 0.0;
   //  const double T_0 = 1.0;
            
   //  q(0) = r_0 + r_0*0.1*sin(2*r_xyz*M_PI*x(0))*sin(2*r_xyz*M_PI*x(1))*sin(2*r_xyz*M_PI*x(2));
   //  q(1) = u_0*((pow(x(0),3)/3. - pow(x(0),2)/2.) + (pow(x(1),3)/3. - pow(x(1),2)/2.) + (pow(x(2),3)/3. - pow(x(2),2)/2.)); 
   //  q(2) = v_0*((pow(x(0),3)/3. - pow(x(0),2)/2.) + (pow(x(1),3)/3. - pow(x(1),2)/2.) + (pow(x(2),3)/3. - pow(x(2),2)/2.)); 
   //  q(3) = w_0*((pow(x(0),3)/3. - pow(x(0),2)/2.) + (pow(x(1),3)/3. - pow(x(1),2)/2.) + (pow(x(2),3)/3. - pow(x(2),2)/2.)); 
   //  double T = T_0;
   //  double p = q(0) * T;
   //  q(4) = p/euler::gami + 0.5 * q(0) * (q(1)*q(1) + q(2)*q(2) + q(3)*q(3));
   q(0) = 1.0;
   q(1) = 0.5;
   q(2) = 0.0;
   q(3) = 0.0;
   q(4) = 1.9107142857142858;
}