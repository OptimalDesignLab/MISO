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
   /// and load elements for implicit/explicit ODE integration
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

/// Class that can handle implicit or explicit time marching of linear or
/// nonlinear ODEs
/// TODO: think about how to handle partial assebmly of residual jacobian and
///       stiffness matrices
class MachEvolver : public mfem::TimeDependentOperator
{
public:
   /// Serves as an base class for linear/nonlinear explicit/implicit time
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

   virtual ~MachEvolver() {}
   
protected:
   /// pointer to mass bilinear form (not owned)
   mfem::OperatorHandle *mass;
   /// pointer to nonlinear form
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

/// For explicit time marching of nonlinear problems
class NonlinearEvolver : public MachEvolver
{
public:
   /// Nonlinear evolver that combines the mass, res, stiff, and load elements
   /// for explicit ODE integration
   /// \param[in] mass - bilinear form for mass matrix (not owned)
   /// \param[in] res - nonlinear residual operator (not owned)
   /// \param[in] stiff - bilinear form for stiffness matrix (not owned)
   /// \param[in] load - load vector (not owned)
   /// \param[in] out - outstream to use pointer (not owned)
   /// \param[in] start_time - time to start integration from
   ///                         (important for time-variant sources)
   /// \param[in] type - solver type; explicit or implicit
   /// \note supports partial assembly of mass matrix
   NonlinearEvolver(BilinearFormType *mass, NonlinearFormType *res,
                    BilinearFormType *stiff = nullptr,
                    mfem::Vector *load = nullptr,
                    std::ostream &outstream = std::cout,
                   double start_time = 0.0)
      : MachEvolver(mass, res, stiff, load, outstream, start_time,
                    EXPLICIT) {};
};

/// For implicit time marching of nonlinear problems
class ImplicitNonlinearEvolver : public MachEvolver
{
public:
   /// Nonlinear evolver that combines the mass, res, stiff, and load elements
   /// for implicit ODE integration
   /// \param[in] mass - bilinear form for mass matrix (not owned)
   /// \param[in] res - nonlinear residual operator (not owned)
   /// \param[in] stiff - bilinear form for stiffness matrix (not owned)
   /// \param[in] load - load vector (not owned)
   /// \param[in] out - outstream to use pointer (not owned)
   /// \param[in] start_time - time to start integration from
   ///                         (important for time-variant sources)
   /// \param[in] type - solver type; explicit or implicit
   /// \note supports partial assembly of mass matrix
   ImplicitNonlinearEvolver(BilinearFormType *mass, NonlinearFormType *res,
                            BilinearFormType *stiff = nullptr,
                            mfem::Vector *load = nullptr,
                            std::ostream &outstream = std::cout,
                            double start_time = 0.0)
      : MachEvolver(mass, res, stiff, load, outstream, start_time,
                    IMPLICIT) {};
};

/// For explicit time marching of linear problems
class LinearEvolver : public MachEvolver
{
public:
   /// Linear evolver that combines the mass, stiff, and load elements
   /// for explicit ODE integration
   /// \param[in] mass - bilinear form for mass matrix (not owned)
   /// \param[in] stiff - bilinear form for stiffness matrix (not owned)
   /// \param[in] load - load vector (not owned)
   /// \param[in] out - outstream to use pointer (not owned)
   /// \param[in] start_time - time to start integration from
   ///                         (important for time-variant sources)
   /// \param[in] type - solver type; explicit or implicit
   /// \note supports partial assembly of mass matrix
   LinearEvolver(BilinearFormType *mass,
                 BilinearFormType *stiff,
                 mfem::Vector *load = nullptr,
                 std::ostream &outstream = std::cout,
                 double start_time = 0.0)
      : MachEvolver(mass, nullptr, stiff, load, outstream, start_time,
                    EXPLICIT) {};
};

/// For implicit time marching of linear problems
class ImplicitLinearEvolver : public MachEvolver
{
public:
   /// Linear evolver that combines the mass, stiff, and load elements
   /// for implicit ODE integration
   /// \param[in] mass - bilinear form for mass matrix (not owned)
   /// \param[in] stiff - bilinear form for stiffness matrix (not owned)
   /// \param[in] load - load vector (not owned)
   /// \param[in] out - outstream to use pointer (not owned)
   /// \param[in] start_time - time to start integration from
   ///                         (important for time-variant sources)
   /// \param[in] type - solver type; explicit or implicit
   /// \note supports partial assembly of mass matrix
   ImplicitLinearEvolver(BilinearFormType *mass,
                         BilinearFormType *stiff,
                         mfem::Vector *load = nullptr,
                         std::ostream &outstream = std::cout,
                         double start_time = 0.0)
      : MachEvolver(mass, nullptr, stiff, load, outstream, start_time,
                    IMPLICIT) {};
};

} // namespace mach

#endif
