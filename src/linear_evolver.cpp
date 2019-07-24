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

LinearEvolver::LinearEvolver(SparseMatrix &mass, SparseMatrix &stiff, ostream &outstream) //, const Vector &_b)
   : out(outstream), TimeDependentOperator(mass.Size()), M(mass), K(stiff), Minv(mass.Size()), z(mass.Size()) //b(_b), z(_M.Size())
{
    // Here we extract the diagonal from the mass matrix and invert it
   #ifdef MFEM_USE_MPI
   int rank;
   MPI_Comm_rank(MPI_COMM_WORLD, &rank);
   #endif
    M.GetDiag(z);
    out << "minimum of z = " << z.Min() << endl;
    out << "maximum of z = " << z.Max() << endl;
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