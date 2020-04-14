#ifndef MACH_LINEAR_EVOLVER
#define MACH_LINEAR_EVOLVER

#include "mfem.hpp"

#include "mach_types.hpp"
#include "inexact_newton.hpp"

/// anonymous namespace
namespace
{

class SystemOperator : public mfem::Operator
{
public:
   /// Nonlinear operator of the form that combines the mass, res, stiff,
   /// and load elements for implicit ODE integration
   /// \param[in] mass - bilinear form for mass matrix (not owned)
   /// \param[in] res - nonlinear residual operator (not owned)
   /// \param[in] stiff - bilinear form for stiffness matrix (not owned)
   /// \param[in] load - load vector (not owned)
   SystemOperator(BilinearFormType *_mass, NonlinearFormType *_res,
                  BilinearFormType *_stiff, mfem::Vector *_load)
      : Operator(_mass->Height()), mass(_mass), res(_res), stiff(_stiff),
        load(_load), dt(0.0), x(nullptr), work(height) {}

   /// Compute r = M@k + R(x + dt*k,t) + K@(x+dt*k) + l
   /// (with `@` denoting matrix-vector multiplication)
   /// \param[in] k - dx/dt 
   /// \param[out] r - the residual
   /// \note the signs on each operator must be accounted for elsewhere
   void Mult(const mfem::Vector &k, mfem::Vector &r) const override
   {
      /// work = x+dt*k = x+dt*dx/dt = x+dx
      add(*x, dt, k, work);
      if (res)
         res->Mult(work, r);
      if (stiff)
         stiff->AddMult(work, r);
      mass->AddMult(k, r);
      if (load)
         r += *load;
   }

   /// Compute J = M + dt * grad(R(x + dt*k, t)) + dt * K
   /// \param[in] k - dx/dt 
   mfem::Operator &GetGradient(const mfem::Vector &k) const override
   {
      MatrixType *jac;
#ifdef MFEM_USE_MPI
      jac = mass->ParallelAssemble();
      if (stiff)
         jac->Add(dt, *(stiff->ParallelAssemble()));
#else
      jac = mass->SpMat();
      if (stiff)
         jac->Add(dt, *(stiff->SpMat()));
#endif
      if (res)
      {
         /// work = x+dt*k = x+dt*dx/dt = x+dx
         add(*x, dt, k, work);
         jac->Add(dt, *dynamic_cast<MatrixType*>(&res->GetGradient(work)));
      } 
      return *jac;
   }

   /// Set current dt and x values - needed to compute action and Jacobian.
   void SetParameters(double _dt, const mfem::Vector *_x)
   {
      dt = _dt;
      x = _x;
   };

private:
   BilinearFormType *mass;
   NonlinearFormType *res;
   BilinearFormType *stiff;
   mfem::Vector *load;

   double dt;
   const mfem::Vector *x;
   mutable mfem::Vector work;
};

}

namespace mach
{

/// TODO: think about how to handle partial assebmly of mass and stiffness
///       matrices
/// TODO: should this take a linear solver pointer during construction?
class MachEvolver : public mfem::TimeDependentOperator
{
public:
   /// Serves as an abstract class for linear/nonlinear explicit/implicit time
   /// marching problems
   /// \param[in] mass - bilinear form for mass matrix (not owned)
   /// \param[in] res - nonlinear residual operator (not owned)
   /// \param[in] stiff - bilinear form for stiffness matrix (not owned)
   /// \param[in] load - load vector (not owned)
   /// \param[in] out - outstream to use pointer (not owned)
   /// \param[in] start_time - time to start integration from
   ///                         (important for time-variant sources)
   /// \param[in] type - solver type; explicit or implicit
   /// \note supports partial assembly of mass matrix
   MachEvolver(BilinearFormType *mass, NonlinearFormType *res,
               BilinearFormType *stiff, mfem::Vector *load,
               std::ostream &outstream, double start_time,
               mfem::TimeDependentOperator::Type type);

   /// Perform the action of the operator: y = k = f(x, t), where k solves
   /// the algebraic equation F(x, k, t) = G(x, t) and t is the current time.
   /// Compute k = M^-1(R(x,t) + Kx + l)
   void Mult(const mfem::Vector &x, mfem::Vector &y) const override;

   /// Solve the implicit equation: k = f(x + dt k, t), for the unknown k at
   /// the current time t. 
   /// Currently implemented for the implicit midpoit method
   void ImplicitSolve(const double dt, const mfem::Vector &x,
                      mfem::Vector &k) override;
   
   /// Set the newton solver to be used for implicit methods
   /// \param[in] newton - pointer to configured newton solver (not owned)
   void SetNewtonSolver(const mfem::NewtonSolver *newton);
   
protected:
   mfem::OperatorHandle *mass;
   NonlinearFormType *res;
   BilinearFormType *stiff;
   mfem::Vector *load;
   std::ostream &out;

   SystemOperator *combined_oper;

   mfem::CGSolver mass_solver;
   std::unique_ptr<mfem::Solver> mass_prec;

   mutable mfem::Vector work;

   mfem::NewtonSolver *newton;

};

// class ImplicitNonlinearEvolver : public MachEvolver
// {
// public:
//    /// class constructor
//    /// \param[in] m - the mass matrix
//    /// \param[in] res - the nonlinear form define the spatial residual
//    /// \param[in] a - set to -1.0 if the spatial residual is on the "wrong" side
//    ImplicitNonlinearEvolver(mfem::NewtonSolver *newton, BilinearFormType *mass,
//                             NonlinearFormType *res,
//                             BilinearFormType *stiff = nullptr,
//                             mfem::Vector *load = nullptr,
//                             std::ostream &outstream = std::cout,
//                             double start_time = 0.0);

//    /// Compute the right-hand side of the ODE system.
//    /// Compute y = f(x + dt * k) - M * k, where k = dx/dt
//    /// \param[in] k - dx/dt
//    /// \param[in/out] y - the residual
//    void Mult(const mfem::Vector &k, mfem::Vector &y) const override;

//    /// Solve the implicit equation: k = f(x + dt k, t), for the unknown k at
//    /// the current time t. 
//    /// Currently implemented for the implicit midpoit method
//    void ImplicitSolve(const double dt, const mfem::Vector &x,
//                       mfem::Vector &k) override;

//    /// Compute the jacobian of implicit evolver: J = dt * f'(x + dt * k) - M
//    /// \param[in] k - dx/dt
//    mfem::Operator &GetGradient(const mfem::Vector &k) const override;

//    /// Set the parameters
//    /// \param[in] dt_ - time step
//    /// \param[in] x_ - current state variable
//    void SetParameters(const double dt_, const mfem::Vector &x_)
//    { 
//       dt = dt_;
//       x = x_;
//    }

//    /// Class destructor
//    virtual ~ImplicitNonlinearEvolver() { }

// protected:
//    mfem::NewtonSolver *newton;


// };

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
                        MatrixType &k, mfem::Vector b, std::ostream &outstream);

   /// Compute explicit solve, if chosen
   virtual void Mult(const mfem::Vector &x, mfem::Vector &k) const;

   /// Implicit solve k = f(q + k * dt, t + dt), where k = dq/dt
   virtual void ImplicitSolve(const double dt, const mfem::Vector &x,
                              mfem::Vector &k);

   /// Implement updates to time dependent terms
   virtual void updateParameters() { }

   /// Class destructor
   virtual ~ImplicitLinearEvolver() { }

protected:
   /// input options
   nlohmann::json options;
   /// linear form (time independent)
   // std::unique_ptr<mfem::LinearForm> force;
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

// /// Implicit Nonlinear evolver
// class ImplicitNonlinearEvolver : public mfem::TimeDependentOperator
// {
// public:
//    /// class constructor
//    /// \param[in] m - the mass matrix
//    /// \param[in] res - the nonlinear form define the spatial residual
//    /// \param[in] a - set to -1.0 if the spatial residual is on the "wrong" side
//    ImplicitNonlinearEvolver(MatrixType &m, NonlinearFormType &r, double a = 1.0);

//    /// Implicit solve k = f(q + k * dt, t + dt), where k = dq/dt
//    /// Currently implemented for the implicit midpoit method
//    virtual void ImplicitSolve(const double dt, const mfem::Vector &x,
//                               mfem::Vector &k);

//    /// Compute y = f(x + dt * k) - M * k, where k = dx/dt
//    /// \param[in] k - dx/dt
//    /// \param[in/out] y - the residual
//    virtual void Mult(const mfem::Vector &k, mfem::Vector &y) const;

//    /// Compute the jacobian of implicit evolver: J = dt * f'(x + dt * k) - M
//    /// \param[in] k - dx/dt
//    virtual mfem::Operator &GetGradient(const mfem::Vector &k) const;

//    /// Set the parameters
//    /// \param[in] dt_ - time step
//    /// \param[in] x_ - current state variable
//    void SetParameters(const double dt_, const mfem::Vector &x_)
//    { 
//       dt = dt_;
//       x = x_;
//    }
//    /// Class destructor
//    virtual ~ImplicitNonlinearEvolver() { }

// private:
//    /// implicit step jacobian
//    //MatrixType *jac;
//    /// used to move the spatial residual to the right-hand-side, if necessary
//    double alpha;
//    /// reference to the mass matrix
//    MatrixType &mass;
//    /// referencee to the nonlinear form i.e. rhs
//    NonlinearFormType &res;
//    /// the time step
//    double dt;
//    /// Vector that hould the current state
//    mfem::Vector x;
//    /// Solver for the implicit time marching
//    std::unique_ptr<mfem::NewtonSolver> newton_solver;
//    //std::unique_ptr<mfem::InexactNewton> newton_solver;
//    /// linear solver in the newton solver
//    std::unique_ptr<mfem::Solver> linear_solver;
//    /// linear system preconditioner for solver in newton solver
//    std::unique_ptr<mfem::Solver> prec;
// };

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
