#include "utils.hpp"

using namespace mfem;
using namespace std;

namespace mach
{

/// performs the Hadamard (elementwise) product: `v(i) = v1(i)*v2(i)`
void multiplyElementwise(const Vector &v1, const Vector &v2, Vector &v)
{
   MFEM_ASSERT( v1.Size() == v2.Size() && v1.Size() == v.Size(), "");
   for (int i = 0; i < v.Size(); ++i)
   {
      v(i) = v1(i)*v2(i);
   }
}

/// performs the Hadamard (elementwise) product: `a(i) *= b(i)`
void multiplyElementwise(const Vector &b, Vector &a)
{
   MFEM_ASSERT( a.Size() == b.Size(), "");
   for (int i = 0; i < a.Size(); ++i)
   {
      a(i) *= b(i);
   }
}

/// performs an elementwise division: `v(i) = v1(i)/v2(i)`
void divideElementwise(const Vector &v1, const Vector &v2, Vector &v)
{
   MFEM_ASSERT( v1.Size() == v2.Size() && v1.Size() == v.Size(), "");
   for (int i = 0; i < v.Size(); ++i)
   {
      v(i) = v1(i)/v2(i);
   }
}

/// performs elementwise inversion: `y(i) = 1/x(i)`
void invertElementwise(const Vector &x, Vector &y)
{
   MFEM_ASSERT( x.Size() == y.Size(), "");
   for (int i = 0; i < x.Size(); ++i)
   {
      y(i) = 1.0/x(i);
   }
}

double quadInterp(double x0, double y0, double dydx0, double x1, double y1)
{
   // Assume the fuction has the form y(x) = c0 + c1 * x + c2 * x^2
   double c0, c1, c2;
   c0 = (dydx0*x0*x0*x1 + y1*x0*x0 - dydx0*x0*x1*x1 - 2*y0*x0*x1 + y0*x1*x1) /
        (x0*x0 - 2*x1*x0 + x1*x1);
   c1 = (2*x0*y0 - 2*x0*y1 - x0*x0*dydx0 + x1*x1*dydx0) / 
        (x0*x0 - 2*x1*x0 + x1*x1);
   c2 = -(y0 - y1 - x0*dydx0 + x1*dydx0) / (x0*x0 - 2*x1*x0 + x1*x1);
   return -c1 / (2*c2);
}

} // namespace mach

