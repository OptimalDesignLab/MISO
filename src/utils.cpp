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

/// performs quadratic interpolation given x0, y0, dy0/dx0, x1, and y1.
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

#ifndef MFEM_USE_LAPACK
   mfem::mfem_error(" Lapack is required for this feature ");
#endif
extern "C" void
dgelss_(int *, int *, int *, double *, int *, double *, int *, double *,
      double *, int *, double *, int *, int *);
void buildInterpolation(int degree, const DenseMatrix &x_center,
    const DenseMatrix &x_quad, DenseMatrix &interp)
{
   // number of quadrature points
   int m = x_quad.Width();
   // number of elements
   int n = x_center.Width();

   // number of rows in little r matrix
   int rows = (degree + 1) * (degree + 2) / 2; 

   // Set the size of interpolation operator
   interp.SetSize(m,n);
   Vector rhs(rows);
   // number of column 
   int nrhs = 1;

   // construct each row of R (also loop over each quadrature point)
   for(int i = 0; i < m; i++)
   {
      // reset the rhs
      rhs = 0.0; rhs(0) = 1.0;
      // construct the aux matrix to solve each row of R
      DenseMatrix r(rows, n);
      r = 0.0;
      // loop over each column of r
      for(int j = 0; j < n; j++)
      {
         double x_diff = x_center(0,j) - x_quad(0,i);
         double y_diff = x_center(1,j) - x_quad(1,i);
         r(0,j) = 1.0;
         int index = 1;
         // loop over different orders
         for(int order = 1; order <= degree; order++)
         {
            for(int c = order; c >= 0; c--)
            {
               r(index, j) = pow(x_diff,c) * pow(y_diff, order-c);
               index++;
            }
         }
      }
      // Solve each row of R and put them back to R
      int info;
      mfem::Vector sv;
      sv.SetSize(std::min(rows, n));
      int rank;
      double rcond = -1.0;
      double *work = NULL;
      double qwork;
      int lwork = -1;
      // query and allocate the optimal workspace
      dgelss_(&rows, &n, &nrhs, r.GetData(), &rows, rhs.GetData(), &rows,
              sv.GetData(), &rcond, &rank, &qwork, &lwork, &info);
      lwork = (int) qwork;
      work = new double [lwork];
      // solve the equation rx = rhs
      dgelss_(&rows, &n, &nrhs, r.GetData(), &rows, rhs.GetData(), &rows,
              sv.GetData(), &rcond, &rank, work, &lwork, &info);
      delete [] work;
      for(int k = 0; k < n; k++)
      {
         interp(i,k) = rhs(k);
      }
   } // end of constructing interp
}

} // namespace mach