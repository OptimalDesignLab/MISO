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

/// Relaxation version of implicit midpoint method
class RRKImplicitMidpointSolver : public mfem::ODESolver
{
public:
   RRKImplicitMidpointSolver(std::ostream *out_stream)
       : mfem::ODESolver(), out(out_stream) {}

   virtual void Init(mfem::TimeDependentOperator &_f);

   virtual void Step(mfem::Vector &x, double &t, double &dt);

protected:
   mfem::Vector k;
   std::ostream *out;
};

class RelaxedNewton: public mfem::NewtonSolver
{
public:
   RelaxedNewton(MPI_Comm comm)
   :  mfem::NewtonSolver(comm), first_iter(true)
   { }

   void SetOperator(const mfem::Operator &op) override;

   /// Solve the nonlinear system with right-hand side @a b.
   /** If `b.Size() != Height()`, then @a b is assumed to be zero. */
   void Mult(const mfem::Vector &b, mfem::Vector &x) const override;

   double ComputeScalingFactor(const mfem::Vector &x,
                               const mfem::Vector &b) const override;

   /// Set the operator used to calculate the energy for the line search
   void SetEnergyOperator(const mfem::NonlinearForm &op);

   void SetLoad(const mfem::Vector *_load) {load = _load;};

protected:
   mutable mfem::Vector rkp1, xkp1;

   const mfem::NonlinearForm *energy;
   const mfem::Vector *load;

   mutable bool first_iter;
};

} // namespace mach

#endif