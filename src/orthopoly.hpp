#ifndef MACH_ORTHOPOLY
#define MACH_ORTHOPOLY

#include "mfem.hpp"

namespace mach
{

   /*!
   * \brief Evaluate a Jacobi polynomial at some points. Based on JacobiP in Hesthaven and
     Warburton's nodal DG book.
   * \param[in] x - points at which to evaluate polynomial - mass matrix
   * \param[in] alpha, beta - define the type of Jacobi Polynomial (alpha + beta != 1)
   * \param[in] N - polynomial degree
   * \param[out] P - the polynomial evaluated at x
   */ 
   void  jacobipoly(const mfem::Vector &x, const double alpha, const double beta,
                            const int N, mfem::Vector &P);

   /*!
   * \brief Evaluate Proriol orthogonal polynomial basis function on the right triangle.
   * \param[in] x, y - locations at which to evaluate the polynomial
   * \param[in] i, j - index pair that defines the basis function to evaluate;
   *  see Hesthaven  and Warburton's Nodal DG book, for example, for a reference.
   * \param[out] P  - basis function at (x , y) 
   */
   void  proriopoly(const mfem::Vector &x, const mfem::Vector &y, const int i, const int j,
                                                mfem::Vector &P);

   /*!
   * \brief  Evaluates lps operator, \f$ I - LL^TH_{k} \f$ as in lps paper, 
   * that projects out modes lower than degree.
   * \param[in] ir - Integration rule based on SBP
   * \param[in] degree  - degree of the polynomial
   * \param[out] lps - lps operator as dense matrix
   */                                          
   void getFilterOperator(const IntegrationRule *ir, const int degree, DenseMatrix &lps);    

   /*!
   * \brief Provides inner product as \f$P P^T w \f$
   * \param[in] P - Matrix for which inner product is done
   * \param[in] w_q - weights of integration points
   * \param[in] N_row - size of w_q
   * \param[in] N_col - # columns of P
   * \param[out] PPtransw - integral inner product
   */
   void integral_inner_prod(Vector &P, Vector &w_q, int N_col, int N_row, Vector &PPtransw);

   /*!
   * \brief Provides Identity matrix
   * \param[in] N_col - # rows/columns
   * \param[out] I - Identity matrix
   */
   void eye(int N_col, Vector &I);                                         
   
} // namespace mach

#endif
