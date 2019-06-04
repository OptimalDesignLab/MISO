#include <fstream>
#include <iostream>
#include "solver.hpp"

using namespace std;
using namespace mfem;

namespace mach
{

AbstractSolver::AbstractSolver(OptionsParser &args)
{
   // references to options here
   //const char *mesh_file = "unitGridTestMesh.msh";
   const char *mesh_file = "periodic-unit-square-tri.mesh"; 
   args.AddOption(&mesh_file, "-m", "--mesh",
                  "Mesh file to use.");
   int degree = 1;
   args.AddOption(&degree, "-d", "--degree",
                  "Degree of the SBP operators.");
   int ode_solver_type = 4;
   args.AddOption(&ode_solver_type, "-s", "--ode-solver",
                  "ODE solver: 0 - Steady Problem,\n\t"
                  "            1 - Forward Euler,\n\t"
                  "            2 - RK2 SSP,\n\t"
                  "            3 - RK3 SSP,\n\t"
                  "            4 - RK4 (default),\n\t"
                  "            6 - RK6.");
   t_final = 10.0;
   args.AddOption(&t_final, "-tf", "--t-final",
                  "Final time; start time is 0.");
   dt = 0.01;
   args.AddOption(&dt, "-dt", "--time-step",
                  "Time step.");               
   args.Parse();
   if (!args.Good())
   {
      args.PrintUsage(cout);
      throw MachException("Invalid command-line option.");
      // TODO: how to handle parallel exits?
   }
   args.PrintOptions(cout);

   // Read the mesh from the given mesh file. We can handle geometrically
   // periodic meshes in this code.
   mesh.reset(new Mesh(mesh_file, 1, 1));
   int dim = mesh->Dimension();
   cout << "problem space dimension = " << dim << endl;

   // Define the ODE solver used for time integration (possibly not used)
   ode_solver = NULL;
   switch (ode_solver_type)
   {
      case 1: ode_solver.reset(new ForwardEulerSolver); break;
      case 2: ode_solver.reset(new RK2Solver(1.0)); break;
      case 3: ode_solver.reset(new RK3SSPSolver); break;
      case 4: ode_solver.reset(new RK4Solver); break;
      case 6: ode_solver.reset(new RK6Solver); break;
      default:
         throw MachException("Unknown ODE solver type " + to_string(ode_solver_type));
         // TODO: parallel exit
   }

   // Refine the mesh here, or have a separate member function?

   // Define the SBP elements and finite-element space; eventually, we will want
   // to have a case or if statement here for both CSBP and DSBP, and (?) standard FEM.
   fec.reset(new C_SBPCollection(degree, dim));
}

AbstractSolver::~AbstractSolver() 
{
   cout << "Deleting Abstract Solver..." << endl;
}

void AbstractSolver::set_initial_condition(
   void (*u_init)(const Vector &, Vector &))
{
   // TODO: Need to verify that this is ok for scalar fields
   VectorFunctionCoefficient u0(num_state, u_init);
   u->ProjectCoefficient(u0);
}

double AbstractSolver::compute_L2_error(
   void (*u_exact)(const Vector &, Vector &))
{
   VectorFunctionCoefficient ue(num_state, u_exact);
   return u->ComputeL2Error(ue);
}

void AbstractSolver::solve_for_state()
{
   // TODO: This is not general enough.

   double t = 0.0;
   evolver->SetTime(t);
   ode_solver->Init(*evolver);

   // output the mesh and initial condition
   // TODO: need to swtich to vtk for SBP
   int precision = 8;
   {
      ofstream omesh("adv.mesh");
      omesh.precision(precision);
      mesh->Print(omesh);
      ofstream osol("adv-init.gf");
      osol.precision(precision);
      u->Save(osol);
   }

   bool done = false;
   for (int ti = 0; !done; )
   {
      double dt_real = min(dt, t_final - t);
      ode_solver->Step(*u, t, dt_real);
      ti++;

      done = (t >= t_final - 1e-8*dt);

/*       if (done || ti % vis_steps == 0)
      {
         cout << "time step: " << ti << ", time: " << t << endl;

         if (visualization)
         {
            sout << "solution\n" << mesh << u << flush;
         }

         if (visit)
         {
            dc->SetCycle(ti);
            dc->SetTime(t);
            dc->Save();
         }
      } */
   }

   // Save the final solution. This output can be viewed later using GLVis:
   // glvis -m unitGridTestMesh.msh -g adv-final.gf".
   {
      ofstream osol("adv-final.gf");
      osol.precision(precision);
      u->Save(osol);
   }
}

} // namespace mach
