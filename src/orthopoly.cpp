#include <assert.h>
#include "orthopoly.hpp"

using namespace std;
using namespace mfem;

namespace mach
{

void jacobiPoly(const Vector &x, const double alpha, const double beta,
                const int degree, Vector &poly)
{
   double gamma0, gamma1, anew, aold, bnew, h1;
   int size = x.Size();
   MFEM_ASSERT( alpha + beta != -1, "");
   MFEM_ASSERT( alpha > -1 && beta > -1, "");
   gamma0 = ((pow(2, alpha + beta + 1)) / (alpha + beta + 1)) *
            (tgamma(alpha + 1) * tgamma(beta + 1) / tgamma(alpha + beta + 1));
   // For degree 0, return a constant function
   Vector poly_0(size);
   poly_0 = 1.0/sqrt(gamma0);
   if (degree == 0)
   {
      poly = poly_0 ;
      return;
   }
   gamma1 = (alpha + 1) * (beta + 1) * gamma0 / (alpha + beta + 3);
   // Set poly_1(:) = ((alpha+beta+2)*x(:) + (alpha-beta))*0.5/sqrt(gamma1)
   Vector poly_1(size);
   poly_1 = alpha + beta;
   add(poly_1, alpha + beta + 2, x, poly_1);
   poly_1 *= 0.5/sqrt(gamma1);
   if (degree == 1)
   {
      poly = poly_1;
      return;
   }
   // Henceforth, poly_0  denotes P_{i} and poly_1 denotes P_{i+1} in recurrence
   aold = (2 / (2 + alpha + beta)) * sqrt((alpha + 1) * (beta + 1) /
                                          (alpha + beta + 3));
   for (int i = 0; i < degree - 1; ++i)
   {
      h1 = 2 * (i + 1) + alpha + beta;
      anew = (2 / (h1 + 2)) * sqrt((i + 2) * (i + 2 + alpha + beta) *
                                   (i + 2 + alpha) * (i + 2 + beta) /
                                   ((h1 + 1) * (h1 + 3)));
      bnew = -((alpha * alpha) - (beta * beta)) / (h1 * (h1 + 2));
      // Rather than using Vector methods and functions, we use a loop here
      // because several actions are performed on each entry
      for (int j = 0; j < size; ++j)
      {
         poly(j) = (1 / anew) * (-aold * poly_0 (j) + (x(j) - bnew) * poly_1(j));
         poly_0 (j) = poly_1(j);
         poly_1(j) = poly(j);
      }
      aold = anew;
   }
}

void prorioPoly(const Vector &x, const Vector &y, const int i, const int j,
                Vector &poly)
{
    int size = x.Size();
    Vector poly_L(size), poly_J(size), xi(size);
    MFEM_ASSERT(i >= 0 && j >= 0, "");
    for (int k = 0; k < size; ++k)
    {
        y(k) != 1.0 ? xi(k) = (2.0*(1 + x(k))/(1 - y(k))) - 1 : xi(k) = -1;
    }
    jacobiPoly(xi, 0.0, 0.0, i, poly_L);
    jacobiPoly(y, 2*i + 1, 0.0, j, poly_J);
    for (int k = 0; k < size; ++k)
    {
        poly(k) = sqrt(2)*poly_L(k)*poly_J(k)*pow(1 - y(k), i);
    }
}

void getFilterOperator(const IntegrationRule *ir, const int degree, DenseMatrix &lps)
{
   int num_nodes = ir->GetNPoints();
   int N = (degree + 1) * (degree + 2) / 2;
   Vector x(num_nodes), y(num_nodes), w(num_nodes);
   for (int i = 0; i < num_nodes; i++)
   {
      const IntegrationPoint &ip = ir->IntPoint(i);
      x(i) = ip.x;
      y(i) = ip.y;
      w(i) = ip.weight;
   }
   // loop over ortho polys up to degree and form Vandermonde matrix
   Vector poly(num_nodes);
   DenseMatrix V(num_nodes, N);
   int ptr = 0;
   for (int r = 0; r <= degree; ++r)
   {
      for (int j = 0; j <= r; ++j)
      {
         V.GetColumnReference(ptr, poly);
         prorioPoly(x, y, r - j, j, poly);
         ptr += 1;
      }
   }
   // Set lps = I - V*V'*H
   MultAAt(V, lps);
   lps.RightScaling(w);
   lps *= -1.0;
   for (int i = 0; i < lps.Size(); ++i)
   {
      lps(i, i) += 1.0;
   }
}

} //namespace mach
