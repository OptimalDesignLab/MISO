#ifndef MFEM_EXTENSIONS
#define MFEM_EXTENSIONS

#include "mfem.hpp"

namespace mach
{
/// Backward Euler pseudo-transient continuation solver
class PseudoTransientSolver : public mfem::ODESolver
{
public:
   PseudoTransientSolver(std::ostream *out_stream) : out(out_stream) { }

   void Init(mfem::TimeDependentOperator &_f) override;

   void Step(mfem::Vector &x, double &t, double &dt) override;

protected:
   mfem::Vector k;
   std::ostream *out;
};

/// Relaxation version of implicit midpoint method
class RRKImplicitMidpointSolver : public mfem::ODESolver
{
public:
   RRKImplicitMidpointSolver(std::ostream *out_stream) : out(out_stream) { }

   void Init(mfem::TimeDependentOperator &_f) override;

   void Step(mfem::Vector &x, double &t, double &dt) override;

protected:
   mfem::Vector k;
   std::ostream *out;
};

}  // namespace mach

#endif
