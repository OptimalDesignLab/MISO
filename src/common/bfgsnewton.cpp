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

void BFGSNewton::SetOperator(const Operator &op)
{
   oper = &op;
   height = op.Height();
   width = op.Width();
   c.SetSize(width);
   x_new.SetSize(width);
}

double BFGSNewton::Mult(mfem::Vector &x, mfem::Vector &opt)
{
   MFEM_ASSERT(oper != NULL, "the Operator is not set (use SetOperator).");

   std::cout << "Beginning of BFGS Newton..." << '\n';
   int numvar = x.Size();
   x_new.SetSize(numvar);
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

   norm0 = norm = oper->GetEnergy(x);
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
      double c_scale = ComputeStepSize(x, b, norm);
      if (c_scale == 0.0)
      {
         converged = 0;
         break;
      }
      c *= (-c_scale);
      // update the state
      x += c;
      // update objective new value and derivative
      norm = oper->GetEnergy(x);

      oper->Mult(x,jac_new);
      // update s,y matrix
      for (int i = 0; i < numvar; i++)
      {
         s(i,0) = c(i);
      }
      y = *(dynamic_cast<DenseMatrix *>(grad_new));
      y -= *(dynamic_cast<DenseMatrix *>(grad));
      // update the derivative
      grad = grad_new;
      
      // update hessian and grad;
      UpdateHessianInverse(ident,s,y);
   }

   final_iter = it;
   final_norm = norm;
}


double BFGSNewton::ComputeStepSize (const Vector &x, const Vector &b,
                                       const double norm)
{
   double s = 1.0;
   // p0, p1, and p0p are used for quadratic interpolation p(s) in [0,1].
   // p0 is the value of p(0), p0p is the derivative p'(0), and 
   // p1 is the value of p(1). */
   double p0, p1, p0p;
   // A temporary vector for calculating p0p.
   Vector temp(r.Size());

   p0 = 0.5 * norm * norm;
   // temp=F'(x_i)*r(x_i)
   jac->Mult(r,temp);
   // c is the negative inexact newton step size.
   p0p = -Dot(c,temp);
   //Calculate the new norm.

   add(x,-1.0,c,x_new);
   oper->Mult(x_new,r);
   const bool have_b = (b.Size()==Height());
   if (have_b)
   {
      r -= b;
   }
   double err_new = Norm(r);

   // Globalization start from here.
   int itt=0;
   while (err_new > (1 - t * (1 - theta) ) * norm)
   {
      p1 = 0.5*err_new*err_new;
      // Quadratic interpolation between [0,1]
      theta = quadInterp(0.0, p0, p0p, 1.0, p1);
      theta = (theta > theta_min) ? theta : theta_min;
      theta = (theta < theta_max) ? theta : theta_max;
      // set the new trial step size. 
      s *= theta;
      // update eta
      eta = 1 - theta * (1- eta);
      eta = (eta < eta_max) ? eta : eta_max;
      // re-evaluate the error norm at new x.
      add(x,-s,c,x_new);
      oper->Mult(x_new, r);
      if (have_b)
      {
         r-=b;
      }
      err_new = Norm(r);

      // Check the iteration counts.
      itt ++;
      if (itt > max_iter)
      {
         mfem::mfem_error("Fail to globalize: Exceed maximum iterations.\n");
         break;
      }
   }
   if (print_level>=0)
   {
      mfem::out << " Globalization factors: theta= "<< s 
            << ", eta= " << eta <<'\n';
   }
   return s;
}

void BFGSNewton::UpdateHessianInverse(const mfem::DenseMatrix &ident,
                                      const mfem::DenseMatrix &s,
                                      const mfem::DenseMatrix &y)
{
   int s = Width();

   DenseMatrix temp1(1);
   MultAtB(y,s,temp1);
   double rho = temp1(0,0);

   DenseMatrix temp2(s);
   MultABt(s,y,temp2);
   temp2 *= rho;
   temp2.Neg();
   temp2 += ident; // (I - rho * s * y')

   DenseMatrix temp3(s);
   MultABt(y,s,temp3);
   temp3 *= rho;
   temp3.Neg();
   temp3 += ident; // (I - rho * y * s')

   DenseMatrix temp4(s);
   Mult(temp2,B,temp4); //  X = (I - rho * s * y') * B
   Mult(temp4,temp3,B) // Y = X * (I - rho * y' * s)

   DenseMatrix temp6(s);
   MultABt(s,s,temp6);
   temp6 *= rho;

   B -= temp6;
}


} // end of namespace mfem
