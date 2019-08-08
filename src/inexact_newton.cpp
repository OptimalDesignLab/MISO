#include "inexact_newton.hpp"

using namespace mfem;
using namespace std;

namespace mfem
{
double InexactNewton::ComputeStepSize (const Vector &x, const Vector &b,
                                       const double norm)
{
   double s = 1.0;
   /* p0, p1, and p0p are used for quadratic interpolation p(s) in [0,1].
      p0 is the value of p(0), p0p is the derivative p'(0), and 
      p1 is the value of p(1). */
   double p0, p1, p0p;
   // A temporary vector for calculating p0p.
   Vector temp(r.Size());

   p0 = 0.5 * norm * norm;
   GetSolver()->Mult(r,temp);
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
   return s;
}

void InexactNewton::SetOperator(const Operator &op)
{
   oper = &op;
   height = op.Height();
   width = op.Width();
   MFEM_ASSERT(height == width, "square Operator is required.");

   r.SetSize(width);
   c.SetSize(width);
   x_new.SetSize(width);
}


void InexactNewton::Mult(const Vector &b, Vector &x)
{
   MFEM_ASSERT(oper != NULL, "the Operator is not set (use SetOperator).");
   MFEM_ASSERT(prec != NULL, "the Solver is not set (use SetSolver).");

   int it;
   double norm0, norm, norm_goal;
   const bool have_b = (b.Size() == Height());
   if (!iterative_mode)
   {
      x = 0.0;
   }
   oper->Mult(x, r);
   if (have_b)
   {
      r -= b;
   }

   norm0 = norm = Norm(r);
   norm_goal = std::max(rel_tol*norm, abs_tol);
   prec->iterative_mode = false;
   static_cast<IterativeSolver*> (prec)->SetRelTol(eta);

   // x_{i+1} = x_i - [DF(x_i)]^{-1} [F(x_i)-b]
   for (it = 0; true; it++)
   {
      MFEM_ASSERT(IsFinite(norm), "norm = " << norm);
      if (print_level >= 0)
      {
         mfem::out << "Inexact Newton iteration " << setw(2) << it
                   << " : ||r|| = " << norm;
         if (it > 0)
         {
            mfem::out << ", ||r||/||r_0|| = " << norm/norm0;
         }
         mfem::out << ". Tolerance for inexact newton solver is " << eta 
            << ".\n";
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
      prec->SetOperator(oper->GetGradient(x));
      prec->Mult(r, c);  // c = [DF(x_i)]^{-1} [F(x_i)-b]
      // mfem::out << ", newton step is " << -c(0) << " and new pos is " 
      //       << x(0)-c(0)<<".\n";
      double c_scale = ComputeStepSize(x, b, norm);
      if(print_level>=0)
      {
         mfem::out << " Globalization scale is " << c_scale << ".\n";
      }      
      if (c_scale == 0.0)
      {
         converged = 0;
         break;
      }
      add(x, -c_scale, c, x);

      oper->Mult(x, r);
      if (have_b)
      {
         r -= b;
      }
      norm = Norm(r);
   }

   final_iter = it;
   final_norm = norm;
}

} // end of namespace
