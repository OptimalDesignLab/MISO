#ifndef MFEM_INEXACTNEWTON
#define MFEM_INEXACTNEWTON

#include "mfem.hpp"
#include "optimization.hpp"

namespace mfem
{


/// min. J(x) with nonlinear equality constraint R(x) = 0 
/// BFGS quasi-newton method with line search 
class BFGSNewtonSolver
{
public:
   BFGSNewtonSolver(double a_init = 1.0, double a_max = 1e3, double cc1 = 1e-4,
                    double cc2 = 0.9, double max = 40);

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

   /// strong wolfe condition variable
   double alpha_init;
   double alpha_max;
   double c1, c2;

   /// Back tracking globalization
   /// \param[in] x - current solution
   /// \param[in] b - the right-hand side vector in `r(x) = b`
   /// \param[in] norm - norm of the current residual, `||r(x)||`
   /// \returns the globalized stepsize
   /// \warning `prec` must be 
   /// \note See Pawlowski et al., doi:10.1137/S0036144504443511 for details
   /// regarding the line search method and its parameters.
   double ComputeStepSize(const mfem::Vector &x, const mfem::Vector &c, const double norm);

   void UpdateHessianInverse(const mfem::Vector &c,
                             const mfem::Vector &jac, const mfem::Vector &jac_new,
                             const mfem::DenseMatrix &I, mfem::DenseMatrix &H);
   
   double Zoom(double alpha_low, double alpha_hi, double phi_low, double phi_init,
               double dphi_init, const mfem::Vector &x, const mfem::Vector &c);

   // double InterpStep(const double &alpha_low, const double &alpha_hi,
   //                   const double &f_low, const double &f_hi,
   //                   const double &df_low, const double &df_hi,
   //                   const bool &deriv_hi);

   // double QuadraticStep(const double &alpha_low, const double &alpha_hi
   //                      const double &f_low, const double &f_hi,
   //                      const double &df_low);
};

} // end of name space mfem

#endif