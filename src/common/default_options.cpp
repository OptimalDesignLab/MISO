#include "default_options.hpp"

namespace mach
{

/// This is placed in its own file because options are likely to grow large.
/// Also, while it would have been nice to use a raw string here to define the
/// default options, this would not have permitted comments.

const nlohmann::json default_options
{   
   {"flow-param", // options related to flow simulations
   {
      {"mach", 0.5}, // far-field mach number
      {"aoa", 0.0},  // far-field angle of attack
      {"roll-axis", 0},  // axis aligned with nose to tail of aircraft
      {"pitch-axis", 1}, // axis in the "vertical" direction
      {"Re", 0.0},  // far-field Reynolds number
      {"Pr", 0.72}, // the Prandtl number
      {"mu", -1.0}   // nondimensional viscosity (if negative, use Sutherland's)
   }},

   {"space-dis", // options related to spatial discretization
   {
      {"degree", 1}, // default operator degree
      {"lps-coeff", 1.0}, // scaling coefficient for local-proj stabilization
      {"basis-type", "csbp"} // csbp & dsbp for continuous & discrete SBP discretization resp. 
   }},

   {"steady", false}, // deprecated; now included in "time-dis"
   {"time-dis", // options related to unsteady time-marching
   {
      {"steady", false}, // if true, solve a steady problem
      {"ode-solver", "RK4"}, // type of ODE solver to use 
      {"const-cfl", false}, // if true, adapt dt to keep cfl constant
      {"t-final", 1.0}, // final time to simulate to
      {"dt", 0.01}, // time-step size when `const-cfl` is false
      {"cfl", 1.0} // target CFL number
   }},

   {"lin-solver",
   {
      {"type", "hypregmres"}, // Default solver
      {"pctype", "hypreeuclid"}, // Default preconditioner type
      {"printlevel", 0}, // linear solver print level (no printing if zero)
      {"maxiter", 100}, // default to 100 iterations
      {"reltol", 1e-12}, // solver relative tolerance
      {"abstol", 1e-12}, // solver absolute tolerance
      {"kdim", 100} // default restart value
   }},

   {"adj-solver",
   {
      {"printlevel", 0}, // adjoint solver print level (no printing if zero)
      {"maxiter", 100}, // maximum number of solver iterations 
      {"reltol", 1e-8}, // adjoint solver relative tolerance
      {"abstol", 1e-10} // adjoint solver absolute tolerance
   }},

   {"petscsolver",
   {
      {"ksptype", "gmres"}, // GMRES is the default solver for PETSC
      {"pctype", "lu"}, // TODO: LU or ILU?
      {"abstol", 1e-10},
      {"reltol", 1e-10},
      {"maxiter", 100},
      {"printlevel", 0}
   }},

   {"newton", // options related to root-finding algorithms
   {
      {"printlevel", 1}, // linear solver print level (no printing if zero)
      {"maxiter", 100}, // default to 100 iterations
      {"reltol", 1e-14}, // solver relative tolerance
      {"abstol", 1e-14}, // solver absolute tolerance
   }},

   {"mesh",
   {
      {"file", "mach.mesh"}, // mesh file name
      {"model-file","mach.dmg"}, // model file name
      {"refine", 0} // recursive uniform refinement; 0 = no refinement
   }},
};

} // namespace mach
