#ifndef MACH_LINEAR_EVOLVER
#define MACH_LINEAR_EVOLVER

#include "mfem.hpp"

#include "mach_types.hpp"
#include "inexact_newton.hpp"

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
   ImplicitLinearEvolver(const std::string &opt_file_name, MatrixType &m, 
                        MatrixType &k, std::unique_ptr<mfem::LinearForm> b, std::ostream &outstream);

   /// Compute explicit solve, if chosen
   virtual void Mult(const mfem::Vector &x, mfem::Vector &k) const;

   /// Implicit solve k = f(q + k * dt, t + dt), where k = dq/dt
   virtual void ImplicitSolve(const double dt, const mfem::Vector &x,
                              mfem::Vector &k);

   /// Implement updates to time dependent terms
   virtual void updateParameters() { }

   /// Class destructor
   virtual ~ImplicitLinearEvolver() { }

   /// Get the last time step's operator
   mfem::HypreParMatrix* GetOperator() {return T;}
protected:
   /// input options
   nlohmann::json options;
   /// linear form (time independent)
   std::unique_ptr<mfem::LinearForm> force;
   /// linear form (w/ time dependent terms if present)
   std::unique_ptr<mfem::LinearForm> rhs;

private:
   /// used to print information
   std::ostream &out;
   /// mass matrix represented as a matrix
   MatrixType &mass;
   /// stiffness matrix represented as a sparse matrix
   MatrixType &stiff;
   /// time operator represented as a matrix
   mfem::HypreParMatrix *T;
   /// preconditioner for implicit system
   std::unique_ptr<SmootherType> t_prec;
   /// solver for the implicit system
   std::unique_ptr<mfem::CGSolver> t_solver;
   /// preconditioner for explicit system
   std::unique_ptr<SmootherType> m_prec;
   /// solver for the explicit system
   std::unique_ptr<mfem::CGSolver> m_solver;
   /// a work vector
   mutable mfem::Vector z;
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
   /// \param[in] k - dx/dt
   /// \param[in/out] y - the residual
   virtual void Mult(const mfem::Vector &k, mfem::Vector &y) const;

   /// Compute the jacobian of implicit evolver: J = dt * f'(x + dt * k) - M
   /// \param[in] k - dx/dt
   virtual mfem::Operator &GetGradient(const mfem::Vector &k) const;

   /// Set the parameters
   /// \param[in] dt_ - time step
   /// \param[in] x_ - current state variable
   void SetParameters(const double dt_, const mfem::Vector &x_)
   { 
      dt = dt_;
      x = x_;
   }
   /// Class destructor
   virtual ~ImplicitNonlinearEvolver() { }

private:
   /// implicit step jacobian
   //MatrixType *jac;
   /// used to move the spatial residual to the right-hand-side, if necessary
   double alpha;
   /// reference to the mass matrix
   MatrixType &mass;
   /// referencee to the nonlinear form i.e. rhs
   NonlinearFormType &res;
   /// the time step
   double dt;
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

// /// For implicit time marching of nonlinear problems 
// class ImplicitOperator : public mfem::Operator
// {
// public:
//    /// construction of the Implicit Operator
//    ImplicitOperator(MatrixType &m, NonlinearFormType &r)
//    {
//       mass = m;
//       res = r;
//    }

//    /// evaluate the F(q) + M dq/dt
//    virtual void Mult(const mfem::Vector &x, mfem::Vector &y) const;

//    /// Get the jacobian of the implicit operator w.r.t dq/dt
//    virtual Operator &GetGradient(const mfem::Vector &x) const;

//    /// set parameters 
//    void SetParameters(double dt_, mfem::Vector &x_)
//    {
//       dt = dt_;
//       x = x_;
//    }
// private:
//    /// referece to the mass matrix
//    MatrixType &mass;
//    /// referce to the nonlinear form
//    NonlinearFormType &res

//    /// Jacobian of the implicit midpoint method
//    MatrixType *jac;

//    /// aux data
//    double dt;
//    mfem::Vector &x; // referece to the current state

// };


} // namespace mach

#endif
