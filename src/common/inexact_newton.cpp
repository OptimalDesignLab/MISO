#include "inexact_newton.hpp"

#include "utils.hpp"

using namespace std;
using namespace mfem;
using namespace mach;

namespace mfem
{
double InexactNewton::ComputeStepSize(const Vector &x,
                                      const Vector &b,
                                      const double norm) const
{
   double theta = 0.0;
   double s = 1.0;
   // p0, p1, and p0p are used for quadratic interpolation p(s) in [0,1].
   // p0 is the value of p(0), p0p is the derivative p'(0), and
   // p1 is the value of p(1). */
   // A temporary vector for calculating p0p.
   Vector temp(r.Size());

   auto p0 = 0.5 * norm * norm;
   // temp=F'(x_i)*r(x_i)
   jac->Mult(r, temp);
   // c is the negative inexact newton step size.
   auto p0p = -Dot(c, temp);
   // Calculate the new norm.

   add(x, -1.0, c, x_new);
   oper->Mult(x_new, r);
   const bool have_b = (b.Size() == Height());
   if (have_b)
   {
      r -= b;
   }
   double err_new = Norm(r);

   // Globalization start from here.
   int itt = 0;
   while (err_new > (1 - t * (1 - theta)) * norm)
   {
      auto p1 = 0.5 * err_new * err_new;
      // Quadratic interpolation between [0,1]
      theta = quadInterp(0.0, p0, p0p, 1.0, p1);
      theta = (theta > theta_min) ? theta : theta_min;
      theta = (theta < theta_max) ? theta : theta_max;
      // set the new trial step size.
      s *= theta;
      // update eta
      eta = 1 - theta * (1 - eta);
      eta = (eta < eta_max) ? eta : eta_max;
      // re-evaluate the error norm at new x.
      add(x, -s, c, x_new);
      oper->Mult(x_new, r);
      if (have_b)
      {
         r -= b;
      }
      err_new = Norm(r);

      // Check the iteration counts.
      itt++;
      if (itt > max_iter)
      {
         mfem::mfem_error("Fail to globalize: Exceed maximum iterations.\n");
         break;
      }
   }
   if (print_level >= 0)
   {
      mfem::out << " Globalization factors: theta= " << s << ", eta= " << eta
                << '\n';
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

void InexactNewton::Mult(const Vector &b, Vector &x) const
{
   MFEM_ASSERT(oper != nullptr, "the Operator is not set (use SetOperator).");
   MFEM_ASSERT(prec != nullptr, "the Solver is not set (use SetSolver).");

   std::cout << "Beginning of inexact Newton..." << std::endl;
   std::cout.flush();

   const bool have_b = (b.Size() == Height());
   if (!iterative_mode)
   {
      x = 0.0;
   }
   oper->Mult(x, r);
   if (have_b)
   {
      std::cout << "What is going on!" << endl;
      r -= b;
   }
   std::cout << "Just before inexact Newton iterations" << std::endl;
   std::cout << "Norm(r) = " << Norm(r) << endl;
   std::cout.flush();

   auto norm = Norm(r);
   auto norm0 = norm;
   auto norm_goal = std::max(rel_tol * norm, abs_tol);
   prec->iterative_mode = false;
   // x_{i+1} = x_i - [DF(x_i)]^{-1} [F(x_i)-b]
   for (int it = 0; true; it++)
   {
      MFEM_ASSERT(IsFinite(norm), "norm = " << norm);
      if (print_level >= 0)
      {
         mfem::out << "Inexact Newton iteration " << setw(2) << it
                   << " : ||r|| = " << norm;
         if (it > 0)
         {
            mfem::out << ", ||r||/||r_0|| = " << norm / norm0;
         }
         mfem::out << '\n';
      }

      if (norm <= norm_goal)
      {
         converged = 1;
         final_iter = it;
         break;
      }

      if (it >= max_iter)
      {
         converged = 0;
         final_iter = it;
         break;
      }

      jac = &oper->GetGradient(x);
      // std::cout << "Get the jacobian matrix.\n";
      prec->SetOperator(*jac);
      // std::cout << "jac is set as one operator.\n";
      prec->Mult(r, c);  // c = [DF(x_i)]^{-1} [F(x_i)-b]
      // std::cout << "Solve for the newton step.\n";
      double c_scale = ComputeStepSize(x, b, norm);

      if (c_scale == 0.0)
      {
         converged = 0;
         final_iter = it;
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

   final_norm = norm;
}

}  // namespace mfem
