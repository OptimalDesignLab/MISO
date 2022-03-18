#include "bfgsnewton.hpp"

using namespace std;
using namespace mfem;

namespace mfem
{

BFGSNewtonSolver::BFGSNewtonSolver(double eta_i, double eta_m,double scale,
                                   int max)
{
   eta = eta_i;
   eta_max = eta_m;
   t = scale;
   max_iter = max;
}

void BFGSNewtonSolver::SetOperator(const Operator &op)
{
   oper = &op;
}

void BFGSNewtonSolver::Mult(Vector &x, Vector &opt)
{
   MFEM_ASSERT(oper != NULL, "the Operator is not set (use SetOperator).");

   std::cout << "Beginning of BFGS Newton..." << '\n';
   numvar = x.Size();
   opt.SetSize(numvar);
   c.SetSize(numvar);
   // initialize the hessian inverse as the identity matrix
   DenseMatrix ident(numvar);
   DenseMatrix s(numvar,1);
   DenseMatrix y(numvar,1);
   B.SetSize(numvar);
   
   // initialize the hessian approximation
   for (int i = 0; i < numvar; i++)
   {
      B(i,i) = 1.0;
      ident(i,i) = 1.0;
   }

   int it;
   double norm0, norm, norm_goal;

   norm0 = norm = dynamic_cast<const NonlinearForm*>(oper)->GetEnergy(x);
   norm_goal = std::max(rel_tol*norm, abs_tol);

   // initialize the jacobian
   oper->Mult(x,jac);

   // x_{i+1} = x_i - [DF(x_i)]^{-1} [F(x_i)-b]
   for (it = 0; true; it++)
   {
      MFEM_ASSERT(IsFinite(norm), "norm = " << norm);
      if (print_level >= 0)
      {
         mfem::out << "BFGS optimization iteration " << setw(2) << it
                   << " : ||J|| = " << norm;
         if (it > 0)
         {
            mfem::out << ", ||J||/||J_0|| = " << norm/norm0;
         }
         mfem::out<<'\n';
      }

      
      if (norm <= norm_goal)
      {
         converged = 1;
         break;
      }
      
      if (it >= max_iter)
      {
         converged = 0;
         break;
      }

      // compute c = B * deriv
      B.Mult(jac, c);
      // compute step size
      double c_scale = ComputeStepSize(x,norm);
      if (c_scale == 0.0)
      {
         converged = 0;
         break;
      }
      c *= (-c_scale);
      // update the state
      x += c;

      // update objective new value and derivative
      norm = dynamic_cast<const NonlinearForm*>(oper)->GetEnergy(x);
      oper->Mult(x,jac_new);

      // update hessian
      UpdateHessianInverse(c,jac,jac_new,ident,B);

      // update jac
      jac = jac_new;
   }
   opt = x;
   final_iter = it;
   final_norm = norm;
}


double BFGSNewtonSolver::ComputeStepSize (const Vector &x, const double norm0)
{
   double phi_old = norm0;
   double dphi_init = jac * c; // deriv' * c

   double alpha_new;
   double phi_new;
   double quad_coeff;
   for (int iter = 0; true; iter++)
   {

      // choose new alpha parameter
      if (0 == iter) { alpha_new = alpha_init; }
      if (quad_coeff > 0.0)
      {
         alpha_new = alpha_old - 0.5 * dphi_old/quad_coeff;
         if ((alpha_new < alpha_old) || (alpha_new > alpha_max))
         {
            alpha_new = std::min(2.0*alpha_old,alpha_max);
         }
      }
      else
      {
         alpha_new = std::min(2.0*alpha_old,alpha_max);
      }


      phi_new = dynamic_cast<const NonlinearForm*>(oper)->GetEnergy(x+alpha*c);

      // check if the step violates the sdc,
      // or when i > 0, new phi is greater than the old, then zoom
      if (  (phi_new > phi_init+suff*alpha_new*dphi_new) || ((i > 0) && (phi_new >= phi_old)) )
      {
         
      }
   
   } // end of iteration
   return 1.0;
}

void BFGSNewtonSolver::UpdateHessianInverse(const Vector &s, const Vector &jac,
                                      const Vector &jac_new,const DenseMatrix &I,
                                      DenseMatrix &H)
{
   Vector y(jac_new);
   y -= jac;

   double rho = 1./(y * s);
   
   DenseMatrix s_mat(numvar,1);
   s_mat.SetCol(0,s);

   DenseMatrix y_mat(numvar,1);
   y_mat.SetCol(0,y);

   DenseMatrix sy_mat(numvar);
   DenseMatrix ys_mat(numvar);
   DenseMatrix ss_mat(numvar);

   ::MultABt(s_mat,y_mat,sy_mat);
   ::MultABt(y_mat,s_mat,ys_mat);
   ::MultABt(s_mat,s_mat,ss_mat);

   sy_mat *= (-rho);
   ys_mat *= (-rho);
   ss_mat *= rho;

   sy_mat += I;
   ys_mat += I;

   DenseMatrix syh(numvar);

   ::Mult(sy_mat,H,syh);
   ::Mult(syh,ys_mat,H);

   H += ss_mat;
}


} // end of namespace mfem
