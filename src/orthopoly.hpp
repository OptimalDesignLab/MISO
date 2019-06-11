#ifndef MACH_ORTHOPOLY
#define MACH_ORTHOPOLY

#include "mfem.hpp"

namespace mach
{

/// Evaluate a Jacobi polynomial at some points.
/// \param[in] x - points at which to evaluate polynomial
/// \param[in] alpha, beta - define Jacobi Polynomial (`alpha` + `beta` != 1)
/// \param[in] degree - polynomial degree
/// \param[out] poly - the polynomial evaluated at x
/// 
/// Based on JacobiP in Hesthaven and Warburton's nodal DG book.
void jacobiPoly(const mfem::Vector &x, const double alpha, const double beta,
                const int degree, mfem::Vector &poly);

/// Evaluate Proriol orthogonal polynomial basis on right triangle.
/// \param[in] x, y - locations at which to evaluate the polynomial
/// \param[in] i, j - index pair that defines the basis function to evaluate
/// \param[out] poly  - basis function at (x , y)
///
/// See Hesthaven  and Warburton's Nodal DG book, for example, for a reference.
void prorioPoly(const mfem::Vector &x, const mfem::Vector &y, const int i,
                const int j, mfem::Vector &poly);

/// Construct the local-projection-stabilization operator.
/// \param[in] ir - Integration rule based on SBP
/// \param[in] degree  - degree of the polynomial
/// \param[out] lps - lps operator as dense matrix
///
/// Builds the LPS operator, \f$ I - LL^TH_{k} \f$, that projects out modes
/// lower than `degree`.
void getFilterOperator(const mfem::IntegrationRule *ir,const int degree,
                       mfem::DenseMatrix &lps);

} // namespace mach

#endif
