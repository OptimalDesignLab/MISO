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
   RBFSpace(mfem::Mesh *m, const mfem::FiniteElementCollection *fec,
            mfem::Array<mfem::Vector> center, int vdim = 1,
            int ordering = mfem::Ordering::byVDIM, int degree = 0);

   /// build the prolongation matrix with RBF
   void buildProlongationMatrix();

protected:
   
   /// number of radial basis function
   int numBasis;
   /// polynomial order
   int polyOrder;
   
   /// location of the basis centers
   mfem::Array<Vector> basisCenter;
   /// the shape parameters
   mfem::Array<DenseMatrix> shapeParam;
   /// selected basis for each element (currently it is fixed upon setup)
   mfem::Array<Vector> selectedBasis;

   /// Initialize the patches/stencil given poly order
   void InitializeStencil();
   /// Initialize the shape parameters
   void InitializeShapeParameter();
};

} // end of namespace mfem
#endif