#ifndef MFEM_LINESEARCH_NEWTON
#define MFEM_LINESEARCH_NEWTON
#include "mfem.hpp"
using namespace std;
namespace mfem
{
/// Newton's method with line search algorithm
class LineSearchNewton : public mfem::NewtonSolver
{
public:
   LineSearchNewton(int maxitr = 10) : maxlsitr(maxitr) { }
   /// Constructor for LineSearchNewton Solver
   /// \param[in] comm - a MPI communicator
   /// \param[in] maxitr - maximum number of linesearch iterations
   LineSearchNewton(MPI_Comm comm, int maxitr = 10)
    : NewtonSolver(comm), maxlsitr(maxitr)
   { }

   /** @brief This method implements a line search algorithm. */
   double ComputeScalingFactor(const Vector &x, const Vector &b) const override
   {
      double alpha = 1.0;
      const bool have_b = (b.Size() == Height());
      oper->Mult(x, r);
      if (have_b)
      {
         r -= b;
      }
      prec->SetOperator(oper->GetGradient(x));
      prec->Mult(r, c);
      double norm0 = Norm(r);
      cout << "||r_0|| " << norm0 << endl;
      double norm = norm0;
      double tol = 1e-13;
      for (int k = 0; k < maxlsitr; ++k)
      {
         x_new = 0.0;
         add(x, -alpha, c, x_new);
         oper->Mult(x_new, r);
         if (have_b)
         {
            r -= b;
         }
         norm = Norm(r);
         mfem::out << "          Linesearch iteration " << setw(2) << k
                   << " : ||r|| = " << norm;
         mfem::out << ", ||r||/||r_0|| = " << norm / norm0;
         mfem::out << '\n';
         if (norm < (tol + norm0))
         {
            return alpha;
         }
         // norm0 = norm;

         alpha *= 0.1;
      }
      MFEM_ABORT("line-search failed to choose alpha !! " << alpha);
   }

   void SetOperator(const Operator &op) override
   {
      oper = &op;
      height = op.Height();
      width = op.Width();
      MFEM_ASSERT(height == width, "square Operator is required.");

      r.SetSize(width);
      c.SetSize(width);
      x_new.SetSize(width);
   }

   void Mult(const Vector &b, Vector &x) const override
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
      norm_goal = std::max(rel_tol * norm, abs_tol);

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
               mfem::out << ", ||r||/||r_0|| = " << norm / norm0;
            }
            mfem::out << '\n';
         }
         Monitor(it, norm, r, x);

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
         const double c_scale = ComputeScalingFactor(x, b);
         cout << "c_scale " << c_scale << endl;
         // const double c_scale = 1.0;
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

   mfem::Solver *GetSolver() { return prec; }
   //    /// Set the linear solver for inverting the Jacobian.
   //    /** This method is equivalent to calling SetPreconditioner(). */
   //    void SetSolver(Solver &solver) { prec = &solver; }
protected:
   int maxlsitr;
   mutable Vector r, c;
   /// member vector saves the new x position.
   mutable mfem::Vector x_new;
};
}  // namespace mfem
#endif