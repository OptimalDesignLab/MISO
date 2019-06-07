#include <assert.h>
#include "orthopoly.hpp"
using namespace std;
using namespace mfem;
namespace mach
{

void integral_inner_prod(Vector &P, Vector &w_q, int N_col, int N_row, Vector &PPtransw)
{
    Vector Ptransw(N_col * N_row);
    int ptr1, ptr2;
    // first get P_transpose*w
    for (int j = 0; j < N_col; ++j)
    {
        for (int k = 0; k < N_row; ++k)
        {
            Ptransw((N_row * j) + k) = P((N_row * j) + k) * w_q(k);
        }
    }
    ptr1 = 0;
    ptr2 = 0;
    // integral product
    for (int k = 0; k < N_row * N_row; ++k)
    {
        if (k % N_row == 0 && k != 0)
        {
            ptr1 = 0;
            ptr2 += 1;
        }
        for (int j = 0; j < N_col; ++j)
        {
            PPtransw(k) += P((j * N_row) + ptr2) * Ptransw((j * N_row) + (ptr1));
        }
        ptr1 += 1;
    }
}

void eye(int N_col, Vector &I)
{
    for (int k = 0; k < N_col * N_col; ++k)
    {
        I(k) = 0.0;
        if (k % (N_col + 1) == 0)
        {
            I(k) = 1.0;
        }
    }
}

void jacobipoly(const Vector &x, const double alpha, const double beta,
                const int N, Vector &P)
{
    double gamma0, gamma1, anew, aold, bnew, h1;
    int size = x.Size();
    assert((alpha + beta) != -1);
    assert(alpha > -1 && beta > -1);
    Vector P_0(size), P_1(size);
    gamma0 = ((pow(2, alpha + beta + 1)) / (alpha + beta + 1)) * (tgamma(alpha + 1) * tgamma(beta + 1) / tgamma(alpha + beta + 1));
    for (int i = 0; i < size; ++i)
    {
        P_0(i) = 1 / sqrt(gamma0);
    }
    if (N == 0)
    {
        for (int i = 0; i < size; ++i)
        {
            P(i) = P_0(i);
        }
        return;
    }
    gamma1 = (alpha + 1) * (beta + 1) * gamma0 / (alpha + beta + 3);
    for (int i = 0; i < size; ++i)
    {
        P_1(i) = 0.5 * ((alpha + beta + 2) * x(i) + (alpha - beta)) / sqrt(gamma1);
    }
    if (N == 1)
    {
        for (int i = 0; i < size; ++i)
        {
            P(i) = P_1(i);
        }
        return;
    }
    // Henceforth, P_0 denotes P_{i} and P_1 denotes P_{i+1}
    // repeat value in recurrence
    aold = (2 / (2 + alpha + beta)) * sqrt((alpha + 1) * (beta + 1) / (alpha + beta + 3));
    for (int i = 0; i < N - 1; ++i)
    {
        h1 = 2 * (i + 1) + alpha + beta;
        anew = (2 / (h1 + 2)) * sqrt((i + 2) * (i + 2 + alpha + beta) * (i + 2 + alpha) * (i + 2 + beta) / ((h1 + 1) * (h1 + 3)));
        bnew = -((alpha * alpha) - (beta * beta)) / (h1 * (h1 + 2));
        for (int j = 0; j < size; ++j)
        {
            P(j) = (1 / anew) * (-aold * P_0(j) + (x(j) - bnew) * P_1(j));
        }
        for (int j = 0; j < size; ++j)
        {
            P_0(j) = P_1(j);
            P_1(j) = P(j);
        }
        aold = anew;
    }
}

void proriopoly(const Vector &x, const Vector &y, const int i, const int j,
                Vector &P)
{
    int size = x.Size();
    Vector PL(size), PJ(size), xi(size);
    assert(i >= 0 && j >= 0);
    for (int k = 0; k < size; ++k)
    {
        y(k) != 1.0 ? xi(k) = (2.0 * (1 + x(k)) / (1 - y(k))) - 1 : xi(k) = -1;
    }
    // get polynomial with alpha, beta=0
    jacobipoly(xi, 0.0, 0.0, i, size, PL);
    // polynomial with non-zero alpha
    jacobipoly(y, 2 * i + 1, 0.0, j, size, PJ);
    for (int k = 0; k < size; ++k)
    {
        P(k) = sqrt(2) * PL(k) * PJ(k) * pow(1 - y(k), i);
    }
}

void getFilterOperator(const IntegrationRule *ir, const int degree, DenseMatrix &lps)
{

    int d, size, N;
    size = ir->GetNPoints();
    Vector x(size), y(size), w(size), P(size);
    d = degree;
    N = (d + 1) * (d + 2) / 2;
    Vector I(size * size), diff(size * size);
    Vector V(size * N), Vvtransw(size * size);
    for (int i = 0; i < size; i++)
    {
        const IntegrationPoint &ip = ir->IntPoint(i);
        x(i) = ip.x;
        y(i) = ip.y;
        w(i) = ip.weight;
    }
    int i;
    int ptr = 0;
    // loop over ortho polys up to degree d
    for (int r = 0; r <= d; ++r)
    {
        for (int j = 0; j <= r; ++j)
        {
            i = r - j;
            proriopoly(x, y, i, j, size, P);
            for (int k = 0; k < size; ++k)
            {
                V(size * ptr + k) = P(k);
            }
            ptr += 1;
        }
    }
    integral_inner_prod(V, w, N, size, Vvtransw);
    eye(size, I); // get I matrix
    for (int k = 0; k < size * size; ++k)
    {
        diff(k) = I(k) - Vvtransw(k);
    }
    for (int k = 0; k < size; ++k)
    {
        for (int j = 0; j < size; ++j)
        {
            lps(j, k) = diff(k * size + j);
        }
    }
}

} //namespace mach
