#ifndef MFEM_EXTENSIONS
#define MFEM_EXTENSIONS

#include "mfem.hpp"

namespace mach
{

/// Backward Euler pseudo-transient continuation solver
class PseudoTransientSolver : public mfem::ODESolver
{
protected:
   mfem::Vector k;

public:
   virtual void Init(mfem::TimeDependentOperator &_f);

   virtual void Step(mfem::Vector &x, double &t, double &dt);
};

/// Relaxation version of implicit midpoint method
class RRKImplicitMidpointSolver : public mfem::ODESolver
{
protected:
   mfem::Vector k;

public:
   virtual void Init(mfem::TimeDependentOperator &_f);

   virtual void Step(mfem::Vector &x, double &t, double &dt);
};

} // namespace mach

#endif