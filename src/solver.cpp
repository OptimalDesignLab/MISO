#include "solver.hpp"

using namespace std;
using namespace mfem;

namespace mach
{

AbstractSolver::AbstractSolver(OptionsParser &args)
{
   // references to options here
   const char *mesh_file = "unitGridTestMesh.msh";
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
   mesh = new Mesh(mesh_file, 1, 1);
   int dim = mesh->Dimension();
   cout << "problem space dimension = " << dim << endl;

   // Define the ODE solver used for time integration (possibly not used)
   ode_solver = NULL;
   switch (ode_solver_type)
   {
      case 1: ode_solver = new ForwardEulerSolver; break;
      case 2: ode_solver = new RK2Solver(1.0); break;
      case 3: ode_solver = new RK3SSPSolver; break;
      case 4: ode_solver = new RK4Solver; break;
      case 6: ode_solver = new RK6Solver; break;
      default:
         throw MachException("Unknown ODE solver type " + to_string(ode_solver_type));
         // TODO: parallel exit
   }

   // Refine the mesh here, or have a separate member function?

   // Define the SBP elements and finite-element space; eventually, we will want
   // to have a case or if statement here for both CSBP and DSBP, and (?) standard FEM.
   fec = new C_SBPCollection(degree, dim);
}

AbstractSolver::~AbstractSolver() 
{
   cout << "Deleting Abstract Solver..." << endl;
   delete u; u = NULL;
   delete mesh; mesh = NULL;
}

void AbstractSolver::set_initial_condition(
   void (*u_init)(const Vector &, Vector &))
{
   // TODO: Need to verify that this is ok for scalar fields
   VectorFunctionCoefficient u0(num_state, u_init);
   u->ProjectCoefficient(u0);
}


} // namespace mach
