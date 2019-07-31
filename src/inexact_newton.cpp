#include "inexact_newton.hpp"

using namespace mfem;
using namespace std;

namespace mfem
{

double InexactNewton::ComputeScalingFactor(const Vector &x, const Vector &b)
{
   double s=1;
   double p0, p1,p0p;
   Vector temp(1);
   
   mfem::out << " Entering inexact newton:\n";
   // Calculate p0
   p0 = 0.5*norm*norm;
   // Calculate dydx0
   GetSolver()->Mult(c,temp);
   //jac->Mult(c, temp);
   p0p = 0.5 * Dot(r, temp);
   // Calculate p1
   add(x,-1.0,c,x2);
   oper->Mult(x2,r);
   double err2 = Norm(r);
   //p1 = 0.5*err2*err2;
   
   int itt = 0;
   while( err2 > (1 - t * (1 - theta) ) * norm )
   {
      p1 = 0.5*err2*err2;
      mfem::out << "  p0,p0p,p1 are " << p0 << ", " << p0p << ", " << p1 <<". ";
      // Quadratic interpolation.
      theta = quadInterp(0.0, p0, p0p, 1.0, p1);
      if(theta < theta_min)
      {
         mfem::out<< " theta set to minimum = " <<  theta;
         theta = theta_min;
      }
      else if(theta > theta_max)
      {
         mfem::out <<" theta set to maximum = " << theta;
         theta = theta_max;
      }
      mfem::out << ". theta is set as " << theta <<".\n";
      // Shorten the step.
      s = theta * s;
      eta = 1 - theta * (1- eta);
      // re-evaluate the error norm at new x.
      add(x,-theta,c,x2);
      oper->Mult(x2, r);
      err2 = Norm(r);
      itt ++;
      if(itt > 10)
      {
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
   r2.SetSize(width);
   x2.SetSize(width);
}


void InexactNewton::Mult(const Vector &b, Vector &x)
{
   MFEM_ASSERT(oper != NULL, "the Operator is not set (use SetOperator).");
   MFEM_ASSERT(prec != NULL, "the Solver is not set (use SetSolver).");

   int it;
   double norm0, norm_goal;
   const bool have_b = (b.Size() == Height());
   if (!iterative_mode)
   {
      //x = 0.0;
   }
   oper->Mult(x, r);
   //mfem::out << "\n Residual is " << r(0);
   // if (have_b)
   // {
   //    r -= b;
   // }

   norm0 = norm = Norm(r);
   norm_goal = std::max(rel_tol*norm, abs_tol);
   prec->iterative_mode = false;

   // x_{i+1} = x_i - [DF(x_i)]^{-1} [F(x_i)-b]
   for (it = 0; true; it++)
   {
      MFEM_ASSERT(IsFinite(norm), "norm = " << norm);
      if (print_level >= 0)
      {
         mfem::out << "Newton iteration " << setw(2) << it
                   << " : ||r|| = " << norm;
         if (it > 0)
         {
            mfem::out << ", ||r||/||r_0|| = " << norm/norm0;
         }
         mfem::out << '\n';
      }
      mfem::out << "x = " << x(0);
      mfem::out << ", b = " << b(0);
      mfem::out << ", norm = " << norm << '\n';
      //mfem::out << "Max iter is" << max_iter <<'\n';

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
      //SetJacobian(oper->GetGradient(x));
      //mfem::out << "In step "<< it <<", gradient operator is set. ";
      static_cast<IterativeSolver*> (prec)->SetRelTol(theta);
      mfem::out << "Tolerance is set as " << theta;
      prec->Mult(r, c);  // c = [DF(x_i)]^{-1} [F(x_i)-b]
      //prec->Mult(x, c);
      mfem::out << ", newton step size is " << c(0) << ".\n";
   
      double c_scale = ComputeScalingFactor(x, b);
      mfem::out << " After inexact newton, scale is " << c_scale << '\n';
      if (c_scale == 0.0)
      {
         converged = 0;
         break;
      }
      add(x, -c_scale, c, x);

      oper->Mult(x, r);
      // if (have_b)
      // {
      //    r -= b;
      // }
      norm = Norm(r);
   }

   final_iter = it;
   final_norm = norm;
}

} // end of namespace
