/// Solve the Navier-Stokes MMS verification

#include "mfem.hpp"
#include "navier_stokes.hpp"
#include <fstream>
#include <iostream>

#include "euler_fluxes.hpp"
#include "euler_integ.hpp"
#include "flow_solver.hpp"

using namespace std;
using namespace mfem;
using namespace mach;

// // Provide the options explicitly for regression tests
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
      "dt": 0.01,
      "cfl": 0.1,
      "res-exp": 1.0
   },
   "nonlin-solver": {
      "printlevel": 1,
      "maxiter": 50,
      "reltol": 1e-6,
      "abstol": 1e-8
   },
   "lin-solver": {
      "type": "hyprefgmres",
      "printlevel": 1,
      "filllevel": 3,
      "maxiter": 100,
      "reltol": 1e-2,
      "abstol": 1e-12
   },
   "lin-prec": {
      "type": "hypreilu",
      "lev-fill": 1
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
unique_ptr<Mesh> buildCurvilinearMesh(int degree, int num_x, int num_y, int num_z);

/// \brief Defines the exact solution for the manufactured solution
/// \param[in] x - coordinate of the point at which the state is needed
/// \param[out] u - state variables stored as a 4-vector
void uexact(const Vector &x, Vector& u);

int main(int argc, char *argv[])
{
   // const char *options_file = "navier_stokes3D_mms_options.json";

   // Initialize MPI
   int num_procs, rank;

   MPI_Init(&argc, &argv);
   MPI_Comm_size(MPI_COMM_WORLD, &num_procs);
   MPI_Comm_rank(MPI_COMM_WORLD, &rank);
   ostream *out = getOutStream(rank);
   *out << std::setprecision(15); 

   // Parse command-line options
   OptionsParser args(argc, argv);

   int degree = 0;
   int nx = 2;
   int ny = 2;
   int nz = 2;
   //args.AddOption(&options_file, "-o", "--options",
   //               "Options file to use.");
   args.AddOption(&degree, "-d", "--degree", "poly. degree of mesh mapping");
   args.AddOption(&nx, "-nx", "--num-x", "number of x-direction segments");
   args.AddOption(&ny, "-ny", "--num-y", "number of y-direction segments");
   args.AddOption(&nz, "-nz", "--num-z", "number of z-direction segments");
   args.Parse();
   if (!args.Good())
   {
      args.PrintUsage(*out);
      return 1;
   }

   try
   {
      // construct the mesh
      // string opt_file_name(options_file);
      auto smesh = buildCurvilinearMesh(degree, nx, ny, nz);
      *out << "Number of elements " << smesh->GetNE() <<'\n';
      ofstream sol_ofs("navier_stokes_mms_mesh.vtk");
      sol_ofs.precision(14);
      smesh->PrintVTK(sol_ofs, 3);

      // construct the solver and set the initial condition
      FlowSolver<3,false> solver(MPI_COMM_WORLD, options, std::move(smesh));
      
      // solver.getStateSize() returns the size of space-discretized state variables
      // use that to assign initial conditions for state_timeVarying variables state_tv
      mfem::Vector state_tv(solver.getStateSize());

      // setting initial condition 
      solver.setState(uexact, state_tv);

      // print residual to see that its non-zero
      // create a _json variable state and assigns it the list of state_variables 
      // which can be used in computing residual
      MachInputs inputs({ {"state", state_tv} });
      *out << "Overall initial residual norm is: " << solver.calcResidualNorm(inputs) << "\n";

      // get the initial drag before time stepping
      solver.createOutput("drag", options["outputs"].at("drag"));
      // find values of state_tv that are at the boundaries mentioned in "drag", to compute output
      double init_drag = solver.calcOutput("drag", inputs);
      *out << "before time stepping, drag is " << init_drag << std::endl;

      // do time stepping and solve for the state_variables to get residual to 0
      inputs = MachInputs({});
      solver.solveForState(inputs, state_tv);
      // save final state 
      auto &state = solver.getState();
      state.distributeSharedDofs(state_tv);
      double l2_error = solver.calcConservativeVarsL2Error(uexact, 0);
      *out << "l2 error = " << l2_error << std::endl;

      // solver->solveForState();
      // solver->printSolution("final", degree+1);
      // double drag = solver->calcOutput("drag");

      // *out << "\n|| rho_h - rho ||_{L^2} = " 
      //           << solver->calcL2Error(uexact, 0) << '\n' << endl;
      // *out << "\ndrag \"error\" = " << drag - 1.6 << endl;
      // *out << "\nfinal residual norm = " << solver->calcResidualNorm() << endl;

      // // TEMP
      // //static_cast<NavierStokesSolver<2>*>(solver.get())->setSolutionError(uexact);
      // //solver->printSolution("error", degree+1);

      // // Solve for and print out the adjoint
      // solver->solveForAdjoint("drag");
      // solver->printAdjoint("adjoint", degree+1);

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

unique_ptr<Mesh> buildCurvilinearMesh(int degree, int num_x, int num_y, int num_z)
{
   Mesh mesh = Mesh::MakeCartesian3D(num_x, num_y, num_z, 
                                     Element::TETRAHEDRON, 1.0, 1.0, 1.0, Ordering::byVDIM);
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
            
   q(0) = r_0 + r_0*0.1*sin(2*r_xyz*M_PI*x(0))*sin(2*r_xyz*M_PI*x(1))*sin(2*r_xyz*M_PI*x(2));
   q(1) = u_0*((pow(x(0),3)/3. - pow(x(0),2)/2.) + (pow(x(1),3)/3. - pow(x(1),2)/2.) + (pow(x(2),3)/3. - pow(x(2),2)/2.)); 
   q(2) = v_0*((pow(x(0),3)/3. - pow(x(0),2)/2.) + (pow(x(1),3)/3. - pow(x(1),2)/2.) + (pow(x(2),3)/3. - pow(x(2),2)/2.)); 
   q(3) = w_0*((pow(x(0),3)/3. - pow(x(0),2)/2.) + (pow(x(1),3)/3. - pow(x(1),2)/2.) + (pow(x(2),3)/3. - pow(x(2),2)/2.)); 
   double T = T_0;
   double p = q(0) * T;
   q(4) = p/euler::gami + 0.5 * q(0) * (q(1)*q(1) + q(2)*q(2) + q(3)*q(3)); 
   q(1) *= q(0);
   q(2) *= q(0);
   q(3) *= q(0);
}
