#include "bfgsnewton.hpp"

using namespace std;
using namespace mfem;

namespace mfem
{

BFGSNewtonSolver::BFGSNewtonSolver(double eta_i, double eta_m,double scale)
{
   eta = eta_i;
   eta_max = eta_m;
   t = scale;
}

void BFGSNewton::SetOperator(const Operator &op)
{
   oper = &op;
   height = op.Height();
   width = op.Width();
   r.SetSize(width);
   c.SetSize(width);
   x_new.SetSize(width);
}

void BFGSNewton::Mult(const Vector &b, Vector &x)
{
   MFEM_ASSERT(oper != NULL, "the Operator is not set (use SetOperator).");
   MFEM_ASSERT(prec != NULL, "the Solver is not set (use SetSolver).");

   std::cout << "Beginning of BFGS Newton..." << '\n';

   // initialize the hessian inverse as the identity matrix
   B.SetSize(Width());
   for (int i = 0; i < Width(); i++)
   {
      B(i,i) = 1.0;
   }

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


}



} // end of namespace mfem
