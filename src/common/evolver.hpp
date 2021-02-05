#ifndef MACH_LINEAR_EVOLVER
#define MACH_LINEAR_EVOLVER

#include "mfem.hpp"

#include "adept.h"
#include "mach_types.hpp"
#include "inexact_newton.hpp"

namespace mach
{

/// For systems that are equipped with a non-increasing entropy function
class EntropyConstrainedOperator : public mfem::TimeDependentOperator
{
public:
   /// Default constructor
   EntropyConstrainedOperator(int n) : TimeDependentOperator(n) {}

   /// Evaluate the entropy functional at the given state
   /// \param[in] state - the state at which to evaluate the entropy
   /// \returns the entropy functional
   virtual double Entropy(const mfem::Vector &state) = 0;

   /// Evaluate the residual weighted by the entropy variables
   /// \param[in] dt - evaluate residual at t+dt
   /// \param[in] state - previous time step state
   /// \param[in] k - the approximate time derivative, `du/dt`
   /// \returns the product `w^T res`
   /// \note `w` and `res` are evaluated at `state + dt*k` and time `t+dt`.
   virtual double EntropyChange(double dt, const mfem::Vector &state, 
                                const mfem::Vector &k) = 0;
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
   /// Class constructor 1.
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
   //std::unique_ptr<mfem::HypreSolver> mass_prec;
   /// solver for the mass matrix
   std::unique_ptr<mfem::CGSolver> mass_solver;
   //std::unique_ptr<mfem::HypreGMRES> mass_solver;
   /// a work vector
   mutable mfem::Vector z;
   /// used to move the spatial residual to the right-hand-side, if necessary
   double alpha;
};

/// Implicit Nonlinear evolver
class ImplicitNonlinearEvolver : public mach::EntropyConstrainedOperator
{
public:
   /// class constructor
   /// \param[in] m - the mass matrix
   /// \param[in] res - nonlinear form that defines the spatial residual
   /// \param[in] a - set to -1.0 if the spatial residual is on the "wrong" side
   ImplicitNonlinearEvolver(MatrixType &m, NonlinearFormType &r, 
                            mach::AbstractSolver *abs, double a = 1.0);

   /// Implicit solve k = f(q + k * dt, t + dt), where k = dq/dt
   /// Currently implemented for the implicit midpoint method
   virtual void ImplicitSolve(const double dt, const mfem::Vector &x,
                              mfem::Vector &k);

   /// Compute y = f(x + dt * k) - M * k, where k = dx/dt
   /// \param[in] k - dx/dt
   /// \param[in/out] y - the residual
   virtual void Mult(const mfem::Vector &k, mfem::Vector &y) const;

   /// Compute the jacobian of implicit evolver: J = dt * f'(x + dt * k) - M
   /// \param[in] k - dx/dt
   virtual mfem::Operator &GetGradient(const mfem::Vector &k) const;

   /// Evaluate the entropy functional at the given state
   /// \param[in] state - the state at which to evaluate the entropy
   /// \returns the entropy functional
   virtual double Entropy(const mfem::Vector &state);

   /// Evaluate the residual weighted by the entropy variables
   /// \param[in] dt - evaluate residual at t+dt
   /// \param[in] state - previous time step state
   /// \param[in] k - the approximate time derivative, `du/dt`
   /// \returns the product `w^T res`
   /// \note `w` and `res` are evaluated at `state + dt*k` and time `t+dt`.
   virtual double EntropyChange(double dt, const mfem::Vector &state, 
                                const mfem::Vector &k);
   
   /// Set the parameters
   /// \param[in] dt_ - time step
   /// \param[in] x_ - current state variable
   void SetParameters(const double dt_, const mfem::Vector &x_)
   { 
      dt = dt_;
      x = x_;
   }
   /// check evolver jacobian
   void checkJacobian(void (*pert_fun)(const mfem::Vector &, mfem::Vector &));

   /// Class destructor
   virtual ~ImplicitNonlinearEvolver() { }

private:
   /// used to move the spatial residual to the right-hand-side, if necessary
   double alpha;
   /// reference to the mass matrix
   MatrixType &mass;
   /// referencee to the nonlinear form i.e. rhs
   NonlinearFormType &res;
   /// the time step
   double dt;
   /// the pointer to the abstract solver
   mach::AbstractSolver *abs_solver;
   /// Vector that hould the current state
   mfem::Vector x;
   /// Solver for the implicit time marching
   std::unique_ptr<mfem::NewtonSolver> newton_solver;
   //std::unique_ptr<mfem::InexactNewton> newton_solver;
   /// linear solver in the newton solver
   std::unique_ptr<mfem::Solver> linear_solver;
   /// linear system preconditioner for solver in newton solver
   std::unique_ptr<mfem::Solver> prec;
};

/// Implicit Nonlinear evolver
class ImplicitNonlinearMassEvolver : public mach::EntropyConstrainedOperator
{
public:
   /// class constructor
   /// \param[in] m - the nonlinearform mass matrix
   /// \param[in] res - nonlinear form that defines the spatial residual
   /// \param[in] a - set to -1.0 if the spatial residual is on the "wrong" side
   ImplicitNonlinearMassEvolver(NonlinearFormType &m, NonlinearFormType &r,
                                NonlinearFormType &e, double a = 1.0);

   /// Implicit solve k = f(q + k * dt, t + dt), where k = dq/dt
   /// Currently implemented for the implicit midpoint method
   virtual void ImplicitSolve(const double dt, const mfem::Vector &x,
                              mfem::Vector &k);

   /// Compute y = f(x + dt * k) - M * k, where k = dx/dt
   /// \param[in] k - dx/dt
   /// \param[in/out] y - the residual
   virtual void Mult(const mfem::Vector &k, mfem::Vector &y) const;

   /// Compute the jacobian of implicit evolver: J = dt * f'(x + dt * k) - M
   /// \param[in] k - dx/dt
   virtual mfem::Operator &GetGradient(const mfem::Vector &k) const;

   /// Evaluate the entropy functional at the given state
   /// \param[in] state - the state at which to evaluate the entropy
   /// \returns the entropy functional
   virtual double Entropy(const mfem::Vector &state);

   /// Evaluate the residual weighted by the entropy variables
   /// \param[in] dt - evaluate residual at t+dt
   /// \param[in] state - previous time step state
   /// \param[in] k - the approximate time derivative, `du/dt`
   /// \returns the product `w^T res`
   /// \note `w` and `res` are evaluated at `state + dt*k` and time `t+dt`.
   virtual double EntropyChange(double dt, const mfem::Vector &state, 
                                const mfem::Vector &k);
   
   /// Set the parameters
   /// \param[in] dt_ - time step
   /// \param[in] x_ - current state variable
   void SetParameters(const double dt_, const mfem::Vector x_)
   { 
      x = x_;
      dt = dt_;
   }
   /// check evolver jacobian
   void checkJacobian(void (*pert_fun)(const mfem::Vector &, mfem::Vector &),
                      const mfem::CentGridFunction &u);

   /// temperal function used for checking operator
   void prininit(const mfem::Vector &u);
   /// Class destructor
   virtual ~ImplicitNonlinearMassEvolver() { }

private:
   /// used to move the spatial residual to the right-hand-side, if necessary
   double alpha;
   /// reference to the mass matrix
   NonlinearFormType &mass;
   /// referencee to the nonlinear form i.e. rhs
   NonlinearFormType &res;
   /// reference to a form for computing the entropy 
   NonlinearFormType &ent;
   /// the time step
   double dt;
   /// Vector that hould the current state
   mfem::CentGridFunction x;
   //mfem::CentGridFunction uc;
   /// Solver for the implicit time marching
   std::unique_ptr<mfem::NewtonSolver> newton_solver;
   //std::unique_ptr<mfem::InexactNewton> newton_solver;
   /// linear solver in the newton solver
   std::unique_ptr<mfem::Solver> linear_solver;
   /// linear system preconditioner for solver in newton solver
   std::unique_ptr<mfem::Solver> prec;
};

} // namespace mach

#endif
