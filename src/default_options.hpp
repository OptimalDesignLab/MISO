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
   {"degree", 1}, // default operator degree
   {"mesh-file", "mach.mesh"}, // mesh file name
   {"ode-solver", "RK4"}, // type of ODE solver choices are
                          // "Steady": for steady-state solves
                          // "RK4": for classical 4th order Runge-Kutta
   {"t-final", 1.0}, // final time
   {"dt", 0.01}, // time step size
   {"mesh", 
   {
      {"refine", 0} // recursive uniform refinement; 0 = no refinement
   }}
};

} // namespace mach

#endif