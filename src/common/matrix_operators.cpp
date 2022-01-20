#include "mfem.hpp"

#include "matrix_operators.hpp"

using namespace std;
using namespace mfem;

namespace mach
{
SumOfOperators::SumOfOperators(double alpha,
                               Operator &oper1,
                               double beta,
                               Operator &oper2)
 : Operator(oper1.Height(), oper1.Width()),
   a(alpha),
   b(beta),
   oper_a(&oper1),
   oper_b(&oper2),
   work_vec(oper1.Width())
{
   if ((oper_a->Height() != oper_b->Height()) ||
       (oper_a->Width() != oper_b->Width()))
   {
      throw MachException("SumOfOperators: Operator sizes are incompatible!\n");
   }
}

void SumOfOperators::Add(double alpha,
                               Operator &oper1,
                               double beta,
                               Operator &oper2)
{
   a = alpha; 
   b = beta; 
   oper_a = &oper1;
   oper_b = &oper2;
}

void SumOfOperators::Mult(const Vector &x, Vector &y) const
{
   y.SetSize(x.Size());
   const double zero = 1e-16;
   if (fabs(a) > zero)
   {
      oper_a->Mult(x, work_vec);
      if (fabs(a - 1.0) > zero)
      {
         work_vec *= a;
      }
   }
   if (fabs(b) > zero)
   {
      oper_b->Mult(x, y);
      if (fabs(b - 1.0) > zero)
      {
         y *= b;
      }
   }
   y += work_vec;
}

}  // namespace mach