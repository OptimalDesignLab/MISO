#ifndef MFEM_RBFSPACE
#define MFEM_RBFSPACE

#include "mfem.hpp"
#include "mach_types.hpp"
namespace mfem
{

class RBFSpace : public mfem::FiniteElementSpace
{
public:
   /// class constructor
   RBFSpace(mfem::Mesh *m, const mfem::FiniteElementCollection *fec, int vdim = 1,
            int ordering = mfem::Ordering::byVDIM, int degree = 0);

protected:
   
   /// number of radial basis function
   int numBasis;
   /// polynomial order
   int polyOrder;
   
   /// location of the basis centers
   mfem::Array<Vector *> basisCenter;
   /// the shape parameters
   mfem::Array<DenseMatrix *> shapeParam;


};

} // end of namespace mfem
#endif