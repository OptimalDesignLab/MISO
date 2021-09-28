#ifndef MACH_ORTHOPOLY
#define MACH_ORTHOPOLY

#include "mfem.hpp"

namespace mach
{
/// computes the LGL quadrature nodes and weights on the interval [-1,1]
/// \param[in] num_nodes - the number of nodes (degree+1)
/// \param[out] x - the location of the nodes, listed in increasing order
/// \param[out] w - the quadrature weights corresponding to x
/// The LGL nodes are the zeros of (1-x^2)*P'_N(x), where P_N(x) denotes the
/// (num_nodes-1)th Legendre polynomial.
/// \see C. Canuto, M. Y. Hussaini, A. Quarteroni, T. A. Tang, "Spectral
/// Methods in Fluid Dynamics," Section 2.3. Springer-Verlag 1987.
void getLobattoQuadrature(const int num_nodes,
                          mfem::Vector &x,
                          mfem::Vector &w);

/// Evaluate a Jacobi polynomial at some points.
/// \param[in] x - points at which to evaluate polynomial
/// \param[in] alpha, beta - define Jacobi Polynomial (`alpha` + `beta` != 1)
/// \param[in] degree - polynomial degree
/// \param[out] poly - the polynomial evaluated at x
///
/// Based on JacobiP in Hesthaven and Warburton's nodal DG book.
void jacobiPoly(const mfem::Vector &x,
                const double alpha,
                const double beta,
                const int degree,
                mfem::Vector &poly);

/// Evaluate Proriol orthogonal polynomial basis on right triangle.
/// \param[in] x, y - locations at which to evaluate the polynomial
/// \param[in] i, j - index pair that defines the basis function to evaluate
/// \param[out] poly  - basis function at (x , y)
///
/// See Hesthaven and Warburton's Nodal DG book, for example, for a reference.
/// \warning the reference triangle is (-1,-1), (1,-1), (-1,1) here.
void prorioPoly(const mfem::Vector &x,
                const mfem::Vector &y,
                const int i,
                const int j,
                mfem::Vector &poly);

/// Constructs the Vandermonde matrix for the segment reference domain
/// \param[in] x - locations at which to evaluate the orthogonal polynomials
/// \param[in] degree - maximum polynomial degree to evaluate the polynomials
/// \param[out] V - the Vandermonde matrix
/// \warning the reference segment is (-1), (1) here.  If you
/// want to use this on `mfem`'s reference triangle, you need to adjust `x` and
/// scale `V` by sqrt(2) afterward.
void getVandermondeForSeg(const mfem::Vector &x,
                          const int degree,
                          mfem::DenseMatrix &V);

/// Constructs the Vandermonde matrix for the triangle reference domain
/// \param[in] x, y - locations at which to evaluate the orthogonal polynomials
/// \param[in] degree - maximum polynomial degree to evaluate the polynomials
/// \param[out] V - the Vandermonde matrix
/// \warning the reference triangle is (-1,-1), (1,-1), (-1,1) here.  If you
/// want to use this on `mfem`'s reference triangle, you need to adjust `x` and
/// `y`, and scale `V` by 2.0 afterward.
void getVandermondeForTri(const mfem::Vector &x,
                          const mfem::Vector &y,
                          const int degree,
                          mfem::DenseMatrix &V);

}  // namespace mach

#endif
