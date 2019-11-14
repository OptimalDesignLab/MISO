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
    
   {"finite-element-dis", // options related to finite element discretization
   {
     {"basis-type", "csbp"}, // csbp & dsbp for continuous & discrete SBP discretization resp. 
   }},     

   {"space-dis", // options related to spatial discretization
   {
      {"degree", 1}, // default operator degree
      {"lps-coeff", 1.0} // scaling coefficient for local-proj stabilization
   }},

   {"steady", false}, // if true, solve a steady problem
   {"time-dis", // options related to unsteady time-marching
   {
      {"ode-solver", "RK4"}, // type of ODE solver to use 
      {"const-cfl", false}, // if true, adapt dt to keep cfl constant
      {"t-final", 1.0}, // final time to simulate to
      {"dt", 0.01}, // time-step size when `const-cfl` is false
      {"cfl", 1.0} // target CFL number
   }},

   {"newton", // options related to root-finding algorithms
   {
   }},

   #ifdef MFEM_USE_PUMI
   {"mesh", // options related to the mesh
   {
      {"file", "mach.smb"}, // mesh file name when using pumi
      {"refine", 0} // recursive uniform refinement; 0 = no refinement
   }},
   #else 
   {"mesh", // options related to the mesh
   {
      {"file", "mach.mesh"}, // mesh file name when not using pumi
      {"refine", 0} // recursive uniform refinement; 0 = no refinement
   }},
   #endif
};

} // namespace mach

#endif
