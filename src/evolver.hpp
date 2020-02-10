#ifndef MACH_LINEAR_EVOLVER
#define MACH_LINEAR_EVOLVER

#include "mfem.hpp"
#include "mach_types.hpp"

namespace mach
{

/// For explicit time marching of linear problems
class LinearEvolver : public mfem::TimeDependentOperator
{
public:
   /// Class constructor.
   /// \param[in] m - mass matrix
   /// \param[in] k - stiffness matrix
   /// \param[in] outstream - for output
   LinearEvolver(MatrixType &m, MatrixType &k, std::ostream &outstream);

   /// Applies the action of the linear-evolution operator on `x`.
   /// \param[in] x - `Vector` that is being multiplied by operator
   /// \param[out] y - resulting `Vector` of the action
   virtual void Mult(const mfem::Vector &x, mfem::Vector &y) const;

   /// Class destructor.
   virtual ~LinearEvolver() { }

private:
   /// used to print information
   std::ostream &out;
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

/// For explicit time marching of nonlinear problems
class NonlinearEvolver : public mfem::TimeDependentOperator
{
public:
   /// Class constructor.
   /// \param[in] m - mass matrix
   /// \param[in] res - defines the spatial residual
   /// \param[in] a - set to -1.0 if the spatial residual is on the "wrong" side
   NonlinearEvolver(MatrixType &m, NonlinearFormType &r, double a = 1.0);

   /// Computes the action of the operator based on `x`.
   /// \param[in] x - `Vector` at which the operator is computed
   /// \param[out] y - resulting `Vector` of the action
   virtual void Mult(const mfem::Vector &x, mfem::Vector &y) const;

   /// Class destructor.
   virtual ~NonlinearEvolver() { }

private:
   /// mass matrix represented as a matrix
   MatrixType &mass;
   /// nonlinear spatial residual
   NonlinearFormType &res;
   /// preconditioner for mass matrix
   SmootherType mass_prec;
   /// solver for the mass matrix
   std::unique_ptr<mfem::CGSolver> mass_solver;
   /// a work vector
   mutable mfem::Vector z;
   /// used to move the spatial residual to the right-hand-side, if necessary
   double alpha;
};

/// For implicit time marching of linear problems
class ImplicitLinearEvolver : public mfem::TimeDependentOperator
{
public:
   /// class constructor
   /// \param[in] m - mass matrix
   /// \param[in] k - stiffness matrix
   /// \param[in] outstream - for output
   ImplicitLinearEvolver(MatrixType &m, MatrixType &k, mfem::LinearForm &b, std::ostream &outstream);

   /// Implicit solve k = f(q + k * dt, t + dt), where k = dq/dt
   /// Currently implemented for the implicit midpoint method
   virtual void ImplicitSolve(const double dt, const mfem::Vector &x,
                              mfem::Vector &k);

   /// Class destructor
   virtual ~ImplicitLinearEvolver() { }

private:
   /// used to print information
   std::ostream &out;
   /// mass matrix represented as a matrix
   MatrixType &mass;
   /// stiffness matrix represented as a sparse matrix
   MatrixType &stiff;
   /// linear form (time independent)
   mfem::LinearForm &force;
   /// linear form (w/ time dependent terms if present)
   mfem::LinearForm &rhs;
   /// time operator represented as a matrix
   mfem::HypreParMatrix *T;
   /// preconditioner for implicit system
   std::unique_ptr<SmootherType> t_prec;
   /// solver for the implicit system
   std::unique_ptr<mfem::CGSolver> t_solver;
   /// a work vector
   mutable mfem::Vector z;
};

} // namespace mach

#endif