#ifndef MFEM_EXTENSIONS
#define MFEM_EXTENSIONS

#include "mfem.hpp"

namespace mach
{

/// Backward Euler pseudo-transient continuation solver
class PseudoTransientSolver : public mfem::ODESolver
{
public:
   PseudoTransientSolver(std::ostream *out_stream)
       : mfem::ODESolver(), out(out_stream) {}

   virtual void Init(mfem::TimeDependentOperator &_f);

   virtual void Step(mfem::Vector &x, double &t, double &dt);

protected:
   mfem::Vector k;
   std::ostream *out;
};
} //namespace mach
