#include "linear_evolver.hpp"

using namespace mfem;
using namespace std;

namespace mach
{

void ElementInv(const Vector &x, Vector &y)
{
   MFEM_ASSERT( x.Size() == y.Size(), "");
   for (int i = 0; i < x.Size(); ++i)
   {
      y(i) = 1.0/x(i);
   }
}

void HadamardProd(const Vector &v1, const Vector &v2, Vector &v)
{
   MFEM_ASSERT( v1.Size() == v2.Size() && v1.Size() == v.Size(), "");
   for (int i = 0; i < v.Size(); ++i)
   {
      v(i) = v1(i)*v2(i);
   }
}

LinearEvolver::LinearEvolver(SparseMatrix &mass, SparseMatrix &stiff) //, const Vector &_b)
   : TimeDependentOperator(mass.Size()), M(mass), K(stiff), Minv(mass.Size()), z(mass.Size()) //b(_b), z(_M.Size())
{
    // Here we extract the diagonal from the mass matrix and invert it
    M.GetDiag(z);
    cout << "minimum of z = " << z.Min() << endl;
    cout << "maximum of z = " << z.Max() << endl;
    ElementInv(z, Minv);
}

void LinearEvolver::Mult(const Vector &x, Vector &y) const
{
   // y = M^{-1} (K x)
   //HadamardProd(Minv, x, y);
   K.Mult(x, z);
   HadamardProd(Minv, z, y);
}

}