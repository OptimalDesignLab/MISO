#include <fstream>
#include <iostream>
#include "solver.hpp"
#include "default_options.hpp"

using namespace std;
using namespace mfem;

namespace mach
{

AbstractSolver::AbstractSolver(const string &opt_file_name)
{
   // Set the options; the defaults are overwritten by the values in the file
   // using the merge_patch method
   options = default_options;
   nlohmann::json file_options;
   ifstream options_file(opt_file_name);
   options_file >> file_options;
   options.merge_patch(file_options);
   cout << setw(3) << options << endl;

   // Read the mesh from the given mesh file
   mesh.reset(new Mesh(options["mesh-file"].get<string>().c_str(), 1, 1));
   int dim = mesh->Dimension();
   cout << "problem space dimension = " << dim << endl;

   // Define the ODE solver used for time integration (possibly not used)
   ode_solver = NULL;
   cout << "ode-solver type = " << options["ode-solver"].get<string>() << endl;
   if (options["ode-solver"].get<string>() == "RK1")
   {
      ode_solver.reset(new ForwardEulerSolver);
   }
   if (options["ode-solver"].get<string>() == "RK4")
   {
      ode_solver.reset(new RK4Solver);
   }
   else
   {
      throw MachException("Unknown ODE solver type " +
                          options["ode-solver"].get<string>());
      // TODO: parallel exit
   }

   // Refine the mesh here, or have a separate member function?
   for (int l = 0; l < options["mesh"]["refine"].get<int>(); l++)
   {
      mesh->UniformRefinement();
   }

   // Define the SBP elements and finite-element space; eventually, we will want
   // to have a case or if statement here for both CSBP and DSBP, and (?) standard FEM.
   fec.reset(new C_SBPCollection(options["degree"].get<int>(), dim));
}

AbstractSolver::~AbstractSolver() 
{
   cout << "Deleting Abstract Solver..." << endl;
}

void AbstractSolver::setInitialCondition(
   void (*u_init)(const Vector &, Vector &))
{
   // TODO: Need to verify that this is ok for scalar fields
   VectorFunctionCoefficient u0(num_state, u_init);
   u->ProjectCoefficient(u0);
}

double AbstractSolver::calcL2Error(
   void (*u_exact)(const Vector &, Vector &))
{
   VectorFunctionCoefficient ue(num_state, u_exact);
   return u->ComputeL2Error(ue);
}

void AbstractSolver::solveForState()
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
   double t_final = options["t-final"].get<double>();
   double dt = options["dt"].get<double>();
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
