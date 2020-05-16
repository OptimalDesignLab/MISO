#include "utils.hpp"

using namespace mfem;
using namespace std;

namespace mach
{

/// performs the Hadamard (elementwise) product: `v(i) = v1(i)*v2(i)`
void multiplyElementwise(const Vector &v1, const Vector &v2, Vector &v)
{
   MFEM_ASSERT(v1.Size() == v2.Size() && v1.Size() == v.Size(), "");
   for (int i = 0; i < v.Size(); ++i)
   {
      v(i) = v1(i) * v2(i);
   }
}

/// performs an elementwise division: `v(i) = v1(i)/v2(i)`
void divideElementwise(const Vector &v1, const Vector &v2, Vector &v)
{
   MFEM_ASSERT(v1.Size() == v2.Size() && v1.Size() == v.Size(), "");
   for (int i = 0; i < v.Size(); ++i)
   {
      v(i) = v1(i) / v2(i);
   }
}

/// performs elementwise inversion: `y(i) = 1/x(i)`
void invertElementwise(const Vector &x, Vector &y)
{
   MFEM_ASSERT(x.Size() == y.Size(), "");
   for (int i = 0; i < x.Size(); ++i)
   {
      y(i) = 1.0 / x(i);
   }
}

/// performs quadratic interpolation given x0, y0, dy0/dx0, x1, and y1.
double quadInterp(double x0, double y0, double dydx0, double x1, double y1)
{
   // Assume the fuction has the form y(x) = c0 + c1 * x + c2 * x^2
   double c0, c1, c2;
   c0 = (dydx0 * x0 * x0 * x1 + y1 * x0 * x0 - dydx0 * x0 * x1 * x1 - 2 * y0 * x0 * x1 + y0 * x1 * x1) /
        (x0 * x0 - 2 * x1 * x0 + x1 * x1);
   c1 = (2 * x0 * y0 - 2 * x0 * y1 - x0 * x0 * dydx0 + x1 * x1 * dydx0) /
        (x0 * x0 - 2 * x1 * x0 + x1 * x1);
   c2 = -(y0 - y1 - x0 * dydx0 + x1 * dydx0) / (x0 * x0 - 2 * x1 * x0 + x1 * x1);
   return -c1 / (2 * c2);
}

double bisection(std::function<double(double)> func, double xl, double xr,
                 double ftol, double xtol, int maxiter)
{
   double fl = func(xl);
   double fr = func(xr);
   if (fl*fr > 0.0)
   {
      cerr << "bisection: func(xl)*func(xr) is positive." << endl;
      throw(-1);
   }
   double xm = 0.5*(xl + xr);
   double fm = func(xm);
   int iter = 0;
   while ( (abs(fm) > ftol) && (abs(xr - xl) > xtol) && (iter < maxiter) )
   {
      iter++;
      //cout << "iter = " << iter << ": f(x) = " << fm << endl;
      if (fm*fl < 0.0)
      {
         xr = xm;
         fr = fm;
      }
      else if (fm*fr < 0.0)
      {
         xl = xm;
         fl = fm;
      }
      else {
         break;
      }
      xm = 0.5*(xl + xr);
      fm = func(xm);
   }
   if (iter >= maxiter)
   {
      cerr << "bisection: failed to find a solution in maxiter." << endl;
      throw(-1);
   }
   return xm;
}

double secant(std::function<double(double)> func, double x1, double x2,
              double ftol, double xtol, int maxiter)
{
   double f1 = func(x1); 
   double f2 = func(x2);
   double x, f;
   if (fabs(f1) < fabs(f2))
   {
      // swap x1 and x2 if the latter gives a smaller value
      x = x2;
      f = f2;
      x2 = x1;
      f2 = f1;
      x1 = x;
      f1 = f;
   }
   x = x2;
   f = f2;
   int iter = 0;
   while ( (fabs(f) > ftol) && (iter < maxiter) )
   {
      ++iter;
      try
      {
         double dx = f2*(x2 - x1)/(f2 - f1);
         x -= dx;
         f = func(x);
         if (fabs(dx) < xtol)
         {
            break;
         }
         x1 = x2;
         f1 = f2;
         x2 = x;
         f2 = f;
      }
      catch(std::exception &exception)
      {
         cerr << "secant: " << exception.what() << endl;
      }
   }
   if (iter > maxiter)
   {
      throw MachException("secant: maximum number of iterations exceeded");
   }
   return x;
}

#ifndef MFEM_USE_LAPACK
void dgelss_(int *, int *, int *, double *, int *, double *, int *, double *,
        double *, int *, double *, int *, int *);
void dgels_(char *, int *, int *, int *, double *, int *, double *, int *, double *,
       int *, int *);
#else
extern "C" void
dgelss_(int *, int *, int *, double *, int *, double *, int *, double *,
        double *, int *, double *, int *, int *);
extern "C" void
dgels_(char *, int *, int *, int *, double *, int *, double *, int *, double *,
       int *, int *);
#endif
/// build the interpolation operator on element patch
/// this function will be moved later
#ifdef MFEM_USE_LAPACK
void buildInterpolation(int dim, int degree, int output,
                        const DenseMatrix &x_center, const DenseMatrix &x_quad,
                        DenseMatrix &interp)
{
   // number of quadrature points
   int num_quad = x_quad.Width();
   // number of elements
   int num_el = x_center.Width();

   // number of row and colomn in r matrix
   int m;
   int n = num_el;
   if (1 == dim)
   {
      m = degree + 1;
   }
   else if (2 == dim)
   {
      m = (degree + 1) * (degree + 2) / 2;
   }
   else
   {
      throw MachException("Other dimension interpolation has not been implemented yet.\n");
   }

   // Set the size of interpolation operator
   interp.SetSize(output, num_el);
   // required by the lapack routine
   mfem::DenseMatrix rhs(n, 1);
   char TRANS = 'N';
   int nrhs = 1;
   int lwork = 2 * m * n;
   double work[lwork];

   // construct each row of R (also loop over each quadrature point)
   for (int i = 0; i < output; i++)
   {
      // reset the rhs
      rhs = 0.0;
      rhs(0, 0) = 1.0;
      // construct the aux matrix to solve each row of R
      DenseMatrix r(m, n);
      r = 0.0;
      // loop over each column of r
      for (int j = 0; j < n; j++)
      {
         if (1 == dim)
         {
            double x_diff = x_center(0, j) - x_quad(0, i);
            r(0, j) = 1.0;
            for (int order = 1; order < m; order++)
            {
               r(order, j) = pow(x_diff, order);
            }
         }
         else if (2 == dim)
         {
            double x_diff = x_center(0, j) - x_quad(0, i);
            double y_diff = x_center(1, j) - x_quad(1, i);
            r(0, j) = 1.0;
            int index = 1;
            // loop over different orders
            for (int order = 1; order <= degree; order++)
            {
               for (int c = order; c >= 0; c--)
               {
                  r(index, j) = pow(x_diff, c) * pow(y_diff, order - c);
                  index++;
               }
            }
         }
         else
         {
            throw MachException("Other dimension interpolation has not been implemented yet.\n");
         }
         
      }
      // Solve each row of R and put them back to R
      int info, rank;
      dgels_(&TRANS, &m, &n, &nrhs, r.GetData(), &m, rhs.GetData(), &n,
             work, &lwork, &info);
      MFEM_ASSERT(info == 0, "Fail to solve the underdetermined system.\n");
      // put each row back to interp
      for (int k = 0; k < n; k++)
      {
         interp(i, k) = rhs(k, 0);
      }
   }
} // end of constructing interp

void buildLSInterpolation(int dim, int degree, const DenseMatrix &x_center,
                          const DenseMatrix &x_quad, DenseMatrix &interp)
{
   // get the number of quadrature points and elements.
   int num_quad = x_quad.Width();
   int num_elem = x_center.Width();

   // number of total polynomial basis functions
   int num_basis = -1;
   if (1 == dim)
   {
      num_basis = degree + 1;
   }
   else if (2 == dim)
   {
      num_basis = (degree + 1) * (degree + 2) / 2;
   }
   else if (3 == dim)
   {
      num_basis = (degree + 1) * (degree + 2) * (degree + 3) / 6;
   }
   else
   {
      throw MachException("buildLSInterpolation: dim must be 3 or less.\n");
   }

   // Construct the generalized Vandermonde matrix
   mfem::DenseMatrix V(num_elem, num_basis);
   if (1 == dim)
   {
      for (int i = 0; i < num_elem; ++i)
      {
         double dx = x_center(0, i) - x_center(0, 0);
         for (int p = 0; p <= degree; ++p)
         {
            V(i,p) = pow(dx, p);
         }
      }
   }
   else if (2 == dim)
   {
      for (int i = 0; i < num_elem; ++i)
      {
         double dx = x_center(0, i) - x_center(0, 0);
         double dy = x_center(1, i) - x_center(1, 0);
         int col = 0;
         for (int p = 0; p <= degree; ++p)
         {
            for (int q = 0; q <= p; ++q)
            {
               V(i, col) = pow(dx, p - q)*pow(dy, q);
               ++col;
            }
         }
      }
   }
   else if (3 == dim)
   {
      for (int i = 0; i < num_elem; ++i)
      {
         double dx = x_center(0, i) - x_center(0, 0);
         double dy = x_center(1, i) - x_center(1, 0);
         double dz = x_center(2, i) - x_center(2, 0);
         int col = 0;
         for (int p = 0; p <= degree; ++p)
         {
            for (int q = 0; q <= p; ++q)
            {
               for (int r = 0; r <= p - q; ++r)
               {
                  V(i, col) = pow(dx, p - q - r)*pow(dy, r)*pow(dz, q);
                  ++col;
               }
            }
         }
      }
   }

   // Set the RHS for the LS problem (it's the identity matrix)
   // This will store the solution, that is, the basis coefficients, hence
   // the name `coeff`
   mfem::DenseMatrix coeff(num_elem, num_elem);
   coeff = 0.0;
   for (int i = 0; i < num_elem; ++i)
   {
      coeff(i,i) = 1.0;
   }

   // Set-up and solve the least-squares problem using LAPACK's dgels
   char TRANS = 'N';
   int info;
   int lwork = 2*num_elem*num_basis;
   double work[lwork];
   dgels_(&TRANS, &num_elem, &num_basis, &num_elem, V.GetData(), &num_elem,
          coeff.GetData(), &num_elem, work, &lwork, &info);
   MFEM_ASSERT(info == 0, "Fail to solve the underdetermined system.\n");

   // Perform matrix-matrix multiplication between basis functions evalauted at
   // quadrature nodes and basis function coefficients.
   interp.SetSize(num_quad, num_elem);
   interp = 0.0;
   if (1 == dim)
   {
      // loop over quadrature points
      for (int j = 0; j < num_quad; ++j)
      {
         double dx = x_quad(0, j) - x_center(0, 0);
         // loop over the element centers
         for (int i = 0; i < num_elem; ++i)
         {
            for (int p = 0; p <= degree; ++p)
            {
               interp(j, i) += pow(dx, p)*coeff(p, i);
            }
         }
      }
   }
   else if (2 == dim)
   {
      // loop over quadrature points
      for (int j = 0; j < num_quad; ++j)
      {
         double dx = x_quad(0, j) - x_center(0, 0);
         double dy = x_quad(1, j) - x_center(1, 0);
         // loop over the element centers
         for (int i = 0; i < num_elem; ++i)
         {
            int col = 0;
            for (int p = 0; p <= degree; ++p)
            {
               for (int q = 0; q <= p; ++q)
               {
                  interp(j, i) += pow(dx, p - q) * pow(dy, q) * coeff(col, i);
                  ++col;
               }
            }
         }
      }
   }
   else if (dim == 3)
   {
      // loop over quadrature points
      for (int j = 0; j < num_quad; ++j)
      {
         double dx = x_quad(0, j) - x_center(0, 0);
         double dy = x_quad(1, j) - x_center(1, 0);
         double dz = x_quad(2, j) - x_center(2, 0);
         // loop over the element centers
         for (int i = 0; i < num_elem; ++i)
         {
            int col = 0;
            for (int p = 0; p <= degree; ++p)
            {
               for (int q = 0; q <= p; ++q)
               {
                  for (int r = 0; r <= p - q; ++r)
                  {
                     interp(j, i) += pow(dx, p - q - r) * pow(dy, r) 
                                       * pow(dz, q) * coeff(col, i);
                     ++col;
                  }
               }
            }
         }
      }
   }
}

#endif

} // namespace mach