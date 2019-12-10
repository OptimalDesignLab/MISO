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

/// For explicit or implicit time marching of nonlinear problems
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

   /// Implicit  solve k = f(x + k * dt, t + dt) for k.
   /// Currently implement for the Implicit Midpoint method
   virtual void ImplicitSolve(const double dt, const mfem::Vector &x,
                              mfem::Vector &k);

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

   /// for implicit solve for dq/dt = F(q,t)
   ImplicitOperator impeuler;

   /// newtonsolver for the implicite time marching
   /// TODO: replace it with inexact newton solver
   std::unique_ptr<mfem::NewtonSolver> newton_solver;
   /// Linear solver in the newton solver
   std::unique_ptr<mfem::HypreGMRES> lin_solv;
};

/// Implicit Nonlinear evolver
class ImplicitNonlinearEvolver : public mfem::TimeDependentOperator
{
public:
   /// class constructor
   /// \param[in] m - the mass matrix
   /// \param[in] res - the nonlinear form define the spatial residual
   /// \param[in] a - set to -1.0 if the spatial residual is on the "wrong" side
   ImplicitNonlinearEvolver(MatrixType &m, NonlinearFormType &r, double a = 1.0);

   /// Implicit solve k = f(q + k * dt, t + dt), where k = dq/dt
   /// Currently implemented for the implicit midpoit method
   virtual void ImplicitSolve(const double dt, const mfem::Vector &x,
                              mfem::Vector &k);

   /// Compute y = f(x + dt * k) - M * k, where k = dx/dt
   /// \param[in] k - the time derivative
   /// \param[in/out] y - the residual
   virtual void Mult(const mfem::Vector &k, mfem::Vector &y);

   /// Compute the jacobian of implicit evolver.
   virtual mfem::Operator &GetGradient(const mfem::Vector &k) const;

   /// Class destructor
   virtual ~ImplicitNonlinearEvolver() { }
private:
   /// implicit step jacobian
   MatrixType *jac;
   /// used to move the spatial residual to the right-hand-side, if necessary
   double alpha;
   /// reference to the mass matrix
   MatrixType &mass;
   /// referencee to the nonlinear form i.e. rhs
   NonlinearFormType &res;
   /// the time step
   double dt;

   /// Solver for the implicit time marching
   std::unique_prt<mfem::NewtonSolver> newton_solve;
   /// linear solver in the newton solver
   std::unique_prt<mfem::HypreGMRES> linear_solver;

};

/// For implicit time marching of nonlinear problems 
class ImplicitOperator : public mfem::Operator
{
public:
   /// construction of the Implicit Operator
   ImplicitOperator(MatrixType &m, NonlinearFormType &r)
   {
      mass = m;
      res = r;
   }

   /// evaluate the F(q) + M dq/dt
   virtual void Mult(const mfem::Vector &x, mfem::Vector &y) const;

   /// Get the jacobian of the implicit operator w.r.t dq/dt
   virtual Operator &GetGradient(const mfem::Vector &x) const;

   /// set parameters 
   void SetParameters(double dt_, mfem::Vector &x_)
   {
      dt = dt_;
      x = x_;
   }
private:
   /// referece to the mass matrix
   MatrixType &mass;
   /// referce to the nonlinear form
   NonlinearFormType &res

   /// Jacobian of the implicit midpoint method
   MatrixType *jac;

   /// aux data
   double dt;
   mfem::Vector &x; // referece to the current state

};


} // namespace mach

#endif