#ifndef MACH_DEFAULT_OPTIONS
#define MACH_DEFAULT_OPTIONS

#include "json.hpp"

namespace mach
{

/// Defines the default options for mach
///
/// This is placed in its own file because it is likely to grow large.  Also,
/// while it would have been nice to use a raw string here to define the default
/// options, this would not have permitted comments.
nlohmann::json default_options =
{
   {"model-file","mach.dmg"}, // model file name

   {"space-dis", // options related to spatial discretization
   {
      {"degree", 1}, // default operator degree
      {"lps-coeff", 1.0} // scaling coefficient for local-proj stabilization
   }},

   {"unsteady", // options related to unsteady time-marching
   {
      {"ode-solver", "RK4"}, // type of ODE solver to use 
      {"const-cfl", false}, // if true, adapt dt to keep cfl constant
      {"t-final", 1.0}, // final time to simulate to
      {"dt", 0.01}, // time-step size when `const-cfl` is false
      {"cfl", 1.0} // target CFL number
   }},

   {"mesh", // options related to the mesh
   {
      {"file", "mach.mesh"}, // mesh file name
      {"refine", 0} // recursive uniform refinement; 0 = no refinement
   }}
};

} // namespace mach

#endif