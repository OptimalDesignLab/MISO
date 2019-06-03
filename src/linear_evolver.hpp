#ifndef MACH_LINEAR_EVOLVER
#define MACH_LINEAR_EVOLVER

#include "mfem.hpp"

namespace mach
{

class Linear_Evolver : public mfem::TimeDependentOperator
{
public:
   /*!
   * \brief class constructor
   * \param[in] M - mass matrix
   * \param[in] K - stiffness matrix
   */
   Linear_Evolver(mfem::SparseMatrix &M, mfem::SparseMatrix &K); //, const Vector &_b);

   /*!
   * \brief Applies the action of the linear-evolution operator on `x`
   * \param[in] x - `Vector` that is being multiplied by operator
   * \param[out] y - resulting `Vector` of the action
   */
   virtual void Mult(const mfem::Vector &x, mfem::Vector &y) const;

   /*!
   * \brief class destructor
   */
   virtual ~Linear_Evolver() { }

private:

   /// mass matrix represented as a sparse matrix
   mfem::SparseMatrix &M;

   /// stiffness matrix represented as a sparse matrix
   mfem::SparseMatrix &K;

   // const Vector &b;

   /// inverse diagonal of `M` stored as a `Vector`
   mfem::Vector Minv;

   /// a work vector
   mutable mfem::Vector z;

};

} // namespace mach

#endif