#include <cassert>

#include "utils.hpp"
#include "orthopoly.hpp"

using namespace std;
using namespace mfem;

namespace mach
{
void getLobattoQuadrature(const int num_nodes, Vector &x, Vector &w)
{
   int degree = num_nodes - 1;
   // Use the Chebyshev-Gauss-Lobatto nodes as an initial guess
   for (int i = 0; i < num_nodes; ++i)
   {
      x(i) = -cos(M_PI * i / degree);
   }
   // P holds the Legendre Vandermonde matrix
   DenseMatrix P(num_nodes);
   Vector p_k;
   Vector p_km1;
   Vector p_kp1;
   // Compute P_{degree} using the recusion relation; compute its first and
   // second derivatives and update x using the Newton-Raphson method
   Vector x_old(num_nodes);
   Vector res(num_nodes);
   x_old = 2.0;
   subtract(x, x_old, res);
   while (res.Normlinf() > std::numeric_limits<double>::epsilon())
   {
      x_old = x;
      P.GetColumnReference(0, p_k);
      p_k = 1.0;
      P.GetColumnReference(1, p_k);
      p_k = x;
      for (int k = 1; k < degree; ++k)
      {
         // P[:,k+1]= ((2k+1)*x.*P[:,k] - k*P[:,k-1])/(k+1)
         P.GetColumnReference(k - 1, p_km1);
         P.GetColumnReference(k, p_k);
         P.GetColumnReference(k + 1, p_kp1);
         multiplyElementwise(x, p_k, p_kp1);
         add((2.0 * k + 1.0) / (k + 1.0), p_kp1, -k / (k + 1.0), p_km1, p_kp1);
      }
      // x = xold - ( x.*P[:,degree]-P[:,degree-1] )./( num_nodes*P[:,degree] )
      P.GetColumnReference(degree, p_kp1);
      P.GetColumnReference(degree - 1, p_k);
      multiplyElementwise(x, p_kp1, x);
      x -= p_k;
      divideElementwise(x, p_kp1, x);
      x *= 1.0 / (num_nodes);
      subtract(x_old, x, x);
      subtract(x, x_old, res);
   }
   // w = 2./(num_nodes*degree*P(:,end).^2))
   P.GetColumnReference(num_nodes - 1, p_kp1);
   multiplyElementwise(p_kp1, p_kp1, w);
   w *= 0.5 * num_nodes * degree;
   invertElementwise(w, w);
}

void jacobiPoly(const Vector &x,
                const double alpha,
                const double beta,
                const int degree,
                Vector &poly)
{
   int size = x.Size();
   MFEM_ASSERT(alpha + beta != -1, "");
   MFEM_ASSERT(alpha > -1 && beta > -1, "");
   auto gamma0 =
       ((pow(2, alpha + beta + 1)) / (alpha + beta + 1)) *
       (tgamma(alpha + 1) * tgamma(beta + 1) / tgamma(alpha + beta + 1));
   // For degree 0, return a constant function
   Vector poly_0(size);
   poly_0 = 1.0 / sqrt(gamma0);
   if (degree == 0)
   {
      poly = poly_0;
      return;
   }
   auto gamma1 = (alpha + 1) * (beta + 1) * gamma0 / (alpha + beta + 3);
   // Set poly_1(:) = ((alpha+beta+2)*x(:) + (alpha-beta))*0.5/sqrt(gamma1)
   Vector poly_1(size);
   poly_1 = alpha + beta;
   add(poly_1, alpha + beta + 2, x, poly_1);
   poly_1 *= 0.5 / sqrt(gamma1);
   if (degree == 1)
   {
      poly = poly_1;
      return;
   }
   // Henceforth, poly_0  denotes P_{i} and poly_1 denotes P_{i+1} in recurrence
   auto aold = (2 / (2 + alpha + beta)) *
               sqrt((alpha + 1) * (beta + 1) / (alpha + beta + 3));
   for (int i = 0; i < degree - 1; ++i)
   {
      auto h1 = 2 * (i + 1) + alpha + beta;
      auto anew = (2 / (h1 + 2)) *
                  sqrt((i + 2) * (i + 2 + alpha + beta) * (i + 2 + alpha) *
                       (i + 2 + beta) / ((h1 + 1) * (h1 + 3)));
      auto bnew = -((alpha * alpha) - (beta * beta)) / (h1 * (h1 + 2));
      // Rather than using Vector methods and functions, we use a loop here
      // because several actions are performed on each entry
      for (int j = 0; j < size; ++j)
      {
         poly(j) = (1 / anew) * (-aold * poly_0(j) + (x(j) - bnew) * poly_1(j));
         poly_0(j) = poly_1(j);
         poly_1(j) = poly(j);
      }
      aold = anew;
   }
}

void prorioPoly(const Vector &x,
                const Vector &y,
                const int i,
                const int j,
                Vector &poly)
{
   int size = x.Size();
   Vector poly_L(size);
   Vector poly_J(size);
   Vector xi(size);
   MFEM_ASSERT(i >= 0 && j >= 0, "");
   for (int k = 0; k < size; ++k)
   {
      y(k) != 1.0 ? xi(k) = (2.0 * (1 + x(k)) / (1 - y(k))) - 1 : xi(k) = -1;
   }
   jacobiPoly(xi, 0.0, 0.0, i, poly_L);
   jacobiPoly(y, 2 * i + 1, 0.0, j, poly_J);
   for (int k = 0; k < size; ++k)
   {
      poly(k) = sqrt(2) * poly_L(k) * poly_J(k) * pow(1 - y(k), i);
   }
}

void getVandermondeForSeg(const Vector &x, const int degree, DenseMatrix &V)
{
   int num_nodes = x.Size();
   int N = degree + 1;
   V.SetSize(num_nodes, N);
   Vector poly;
   int ptr = 0;
   for (int r = 0; r <= degree; ++r)
   {
      V.GetColumnReference(ptr, poly);
      mach::jacobiPoly(x, 0.0, 0.0, r, poly);
      ptr += 1;
   }
}

void getVandermondeForTri(const Vector &x,
                          const Vector &y,
                          const int degree,
                          DenseMatrix &V)
{
   MFEM_ASSERT(x.Size() == y.Size(), "");
   int num_nodes = x.Size();
   int N = (degree + 1) * (degree + 2) / 2;
   V.SetSize(num_nodes, N);
   Vector poly;
   int ptr = 0;
   for (int r = 0; r <= degree; ++r)
   {
      for (int j = 0; j <= r; ++j)
      {
         V.GetColumnReference(ptr, poly);
         mach::prorioPoly(x, y, r - j, j, poly);
         ptr += 1;
      }
   }
}

}  // namespace mach
