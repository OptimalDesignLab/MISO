#ifndef MFEM_INEXACTNEWTON
#define MFEM_INEXACTNEWTON

#include "mfem.hpp"

#include "solver.hpp"

using namespace mach;

namespace mfem
{
/// Inexact Newton's method solving F(x) = b with globalization.
class InexactNewton : public mfem::NewtonSolver
{ 
public:
   /// Constructor for Inexact Newton Solver
   /// \param[in] eta_init - initial value of eta, the forcing parameter
   /// \param[in] eta_maximum - maximum value of eta
   /// \param[in] ared_scale - defines target actual reduction in the residual 
   /// \note this only defines the inexact Newton parameters; the actual
   /// problem is defined by the operator `oper`
   /// \note parameters is set up in function `init`
   InexactNewton(double eta_init = 1e-4,double eta_maximum = 1e-1,
                  double ared_scale=1e-4)
   {
      init(eta_init, eta_maximum, ared_scale);
   }

#ifdef MFEM_USE_MPI
   /// Constructor for Inexact Newton Solver
   /// \param[in] comm - a MPI communicator
   /// \param[in] eta_init - initial value of eta, the forcing parameter
   /// \param[in] eta_maximum - maximum value of eta
   /// \param[in] ared_scale - defines target actual reduction in the residual
   /// \note this only defines the inexact Newton parameters; the actual
   /// problem is defined by the operator `oper`
   /// \note parameters is set up in function `init`
   InexactNewton(MPI_Comm comm, double eta_init = 1e-4, 
               double eta_maximum = 1e-1, double ared_scale = 1e-4)
      :NewtonSolver(comm)
   {
      init(eta_init, eta_maximum, ared_scale);
   }
#endif

   /// Set the operator that defines the nonlinear system
   /// \param[in] op - problem operator `r` in `r(x) = b`
   virtual void SetOperator(const mfem::Operator &op);

   /// Solve the nonlinear system with right-hand side b
   /// \param[in] b - the right-hand side vector (can be zero)
   /// \param[in] x - intial "guess" for solution
   virtual void Mult(const mfem::Vector &b, mfem::Vector &x);

   mfem::Solver *GetSolver(){return prec;}

protected:
   /// Jacobian of the nonlinear operator; needed by ComputeStepSize();
   Operator *jac;
   /// member vector saves the new x position.
   mutable mfem::Vector x_new;
   /// Parameters for inexact newton method.
   double theta, eta, eta_max, t;
   const double theta_min = 0.1;
   const double theta_max = 0.5;

private:

   /// Back tracking globalization
   /// \param[in] x - current solution
   /// \param[in] b - the right-hand side vector in `r(x) = b`
   /// \param[in] norm - norm of the current residual, `||r(x)||`
   /// \returns the globalized stepsize
   /// \warning `prec` must be 
   /// \note See Pawlowski et al., doi:10.1137/S0036144504443511 for details
   /// regarding the line search method and its parameters.
   double ComputeStepSize(const mfem::Vector &x, const mfem::Vector &b, 
                        const double norm);

   /// Inexact newton method parameters set up, called in other constructors
   /// \param[in] eta_init - initial value of eta, the forcing parameter
   /// \param[in] eta_maximum - maximum value of eta
   /// \param[in] ared_scale - defines target actual reduction in the residual 
   void init(double eta_init, double eta_maximum, double ared_scale);
};

}

#endif
