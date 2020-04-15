#ifndef MACH_LINEAR_EVOLVER
#define MACH_LINEAR_EVOLVER

#include "mfem.hpp"

#include "mach_types.hpp"
#include "inexact_newton.hpp"

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
               double alpha, std::ostream &outstream, double start_time,
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
   void SetNewtonSolver(mfem::NewtonSolver *newton);

   /// Return a reference to the Jacobian of the combined operator
   /// \param[in] x - the current state
   mfem::Operator &GetGradient(const mfem::Vector &x) const override;

   virtual ~MachEvolver();
   
protected:
   /// pointer to mass bilinear form (not owned)
   mfem::OperatorHandle mass;
   /// pointer to nonlinear form (not owned)
   NonlinearFormType *res;
   /// pointer to stiffness bilinear form (not owned)
   BilinearFormType *stiff;
   ///pointer to load vector (not owned)
   mfem::Vector *load;
   /// used to move the spatial residual to the right-hand-side, if necessary
   double alpha;
   /// outstream for printing
   std::ostream &out;
   /// solver for inverting mass matrix for explicit solves
   /// \note supports partially assembled mass bilinear form
   mfem::CGSolver mass_solver;
   /// preconditioner for inverting mass matrix
   std::unique_ptr<mfem::Solver> mass_prec;
   /// Newton solver for implicit problems (not owned)
   mfem::NewtonSolver *newton;

   /// pointer-to-implementation idiom
   /// Hides implementation details of this operator, and because it's private,
   /// it doesn't pollute the mach namespace
   class SystemOperator;
   /// Operator that combines the linear/nonlinear spatial discretization with
   /// the load vector into one operator used for implicit solves
   std::unique_ptr<SystemOperator> combined_oper;

   /// work vector
   mutable mfem::Vector work;

   /// sets the state and dt for the combined operator
   /// \param[in] dt - time increment
   /// \param[in] x - the current state
   void setOperParameters(double dt, const mfem::Vector *x);
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
                    double alpha = -1.0,
                    BilinearFormType *stiff = nullptr,
                    mfem::Vector *load = nullptr,
                    std::ostream &outstream = std::cout,
                   double start_time = 0.0)
      : MachEvolver(mass, res, stiff, load, alpha, outstream, start_time,
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
                            double alpha = -1.0,
                            BilinearFormType *stiff = nullptr,
                            mfem::Vector *load = nullptr,
                            std::ostream &outstream = std::cout,
                            double start_time = 0.0)
      : MachEvolver(mass, res, stiff, load, alpha, outstream, start_time,
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
                 double alpha = -1.0,
                 mfem::Vector *load = nullptr,
                 std::ostream &outstream = std::cout,
                 double start_time = 0.0)
      : MachEvolver(mass, nullptr, stiff, load, alpha, outstream, start_time,
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
                         double alpha = -1.0,
                         mfem::Vector *load = nullptr,
                         std::ostream &outstream = std::cout,
                         double start_time = 0.0)
      : MachEvolver(mass, nullptr, stiff, load, alpha, outstream, start_time,
                    IMPLICIT) {};
};

} // namespace mach

#endif
