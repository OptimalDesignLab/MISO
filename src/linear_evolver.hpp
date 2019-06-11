#ifndef MACH_LINEAR_EVOLVER
#define MACH_LINEAR_EVOLVER

#include "mfem.hpp"

namespace mach
{

/// For explicit time marching of linear problems (Also nonlinear?)
class LinearEvolver : public mfem::TimeDependentOperator
{
public:
   /// Class constructor.
   /// \param[in] M - mass matrix
   /// \param[in] K - stiffness matrix
   LinearEvolver(mfem::SparseMatrix &M, mfem::SparseMatrix &K); //, const Vector &_b);

   /// Applies the action of the linear-evolution operator on `x`.
   /// \param[in] x - `Vector` that is being multiplied by operator
   /// \param[out] y - resulting `Vector` of the action
   virtual void Mult(const mfem::Vector &x, mfem::Vector &y) const;

   /// Class destructor.
   virtual ~LinearEvolver() { }

private:
   /// mass matrix represented as a sparse matrix
   mfem::SparseMatrix &M;
   /// stiffness matrix represented as a sparse matrix
   mfem::SparseMatrix &K;
   // const Vector &b;
   /// inverse diagonal of `M` stored as a `Vector`
   mfem::Vector Minv;
   /// a work vector
   mutable mfem::Vector z;

};

} // namespace mach

#endif