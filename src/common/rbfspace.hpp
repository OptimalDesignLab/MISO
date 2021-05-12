#ifndef MFEM_RBFSPACE
#define MFEM_RBFSPACE

#include "mfem.hpp"

namespace mfem
{

/// class for declaration for the Radial basis function space
/// The first version have radial basis function centers at
///   element centers
class RBFSpace : public mfem::FiniteElementSpace
{

public:
   RBFSpace(mfem::mesh *m, const mfem::FiniteElementCollection *f,
            int nb, int vdim = 1; int ordering = mfem::Ordering::byVDIM);
protected:
   /// number of Radial functions basis
   int num_basis;

   /// Rhe range that RBF span
   /// This could be an array that holding difference values
   Array<double> span;

   
};

} // end of namespace mfem 
#endif
