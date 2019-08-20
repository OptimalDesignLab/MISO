#ifndef MACH_FUNCTIONAL
#define MACH_FUNCTIONAL

#include "mfem.hpp"
#include "solver.hpp"

namespace mach
{
class FunctionalIntegrator : public mfem::NonlinearIntegrator
{

};

} // namespace mach

#endif