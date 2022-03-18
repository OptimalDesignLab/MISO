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
   BFGSNewtonSolver(double eta_init = 1e-4, double eta_maximum = 1e-1,
              double ared_scale = 1e-4, int max = 30);

   /// Set the operator that defines the nonlinear system
   /// \param[in] op - problem operator `r` in `r(x) = b`
   void SetOperator(const mfem::Operator &op);

   void Mult(mfem::Vector &x, mfem::Vector &opt);

protected:
   int numvar;

   /// the hessian inverse approximation
   mfem::DenseMatrix B;
   mfem::Vector jac;
   mfem::Vector jac_new;
   const mfem::Operator *oper; 

   /// BFGS newton method variable
   double rel_tol,abs_tol;
   bool converged;
   double final_norm;
   int print_level, final_iter, max_iter;
   mutable mfem::Vector c;

   /// strong wolfe condition variable
   


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
   double ComputeStepSize(const mfem::Vector &x, const double norm);
   void UpdateHessianInverse(const mfem::Vector &c,
                             const mfem::Vector &jac, const mfem::Vector &jac_new,
                             const mfem::DenseMatrix &I, mfem::DenseMatrix &H);
};

} // end of name space mfem

#endif