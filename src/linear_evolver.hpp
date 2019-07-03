#ifndef MACH_LINEAR_EVOLVER
#define MACH_LINEAR_EVOLVER

#include "mfem.hpp"
#include "mach_types.hpp"

namespace mach
{

/// For explicit time marching of linear problems (Also nonlinear?)
class LinearEvolver : public mfem::TimeDependentOperator
{
public:
   /// Class constructor.
   /// \param[in] m - mass matrix
   /// \param[in] k - stiffness matrix
   LinearEvolver(MatrixType &m, MatrixType &k); //, const Vector &_b);

   /// Applies the action of the linear-evolution operator on `x`.
   /// \param[in] x - `Vector` that is being multiplied by operator
   /// \param[out] y - resulting `Vector` of the action
   virtual void Mult(const mfem::Vector &x, mfem::Vector &y) const;

   /// Class destructor.
   virtual ~LinearEvolver() { }

private:
   /// mass matrix represented as a matrix
   MatrixType &mass;
   /// stiffness matrix represented as a sparse matrix
   MatrixType &stiff;
   /// preconditioner for mass matrix
   SmootherType mass_prec;
   /// solver for the mass matrix
   std::unique_ptr<mfem::CGSolver> mass_solver;
   /// a work vector
   mutable mfem::Vector z;

};

} // namespace mach

#endif