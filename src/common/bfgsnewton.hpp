#ifndef MFEM_INEXACTNEWTON
#define MFEM_INEXACTNEWTON

#include "mfem.hpp"

namespace mfem
{


/// min. J(x) with nonlinear equality constraint R(x) = 0 
/// BFGS quasi-newton method with line search 
class BFGSNewtonSolver
{
public:
   BFGSNewton(double eta_init = 1e-4, double eta_maximum = 1e-1,
              double ared_scale = 1e-4);

   /// Set the operator that defines the nonlinear system
   /// \param[in] op - problem operator `r` in `r(x) = b`
   void SetOperator(const mfem::Operator &op);

   /// solve the optimization problem
   void Optimize();

protected:
   /// the hessian inverse approximation
   mfem::DenseMatrix B;
   mfem::Operator *jac;
   mfem::Operator *jac_new;
   mfem::Operator *oper; 

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
   void UpdateHessianInverse(const mfem::DenseMatrix &ident,
                             const mfem::DenseMatrix &s,
                             const mfem::DenseMatrix &y);
};

} // end of name space mfem

#endif