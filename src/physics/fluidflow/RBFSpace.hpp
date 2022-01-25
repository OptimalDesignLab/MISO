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
            mfem::Array<Vector*> center, int vdim = 1,
            int ordering = mfem::Ordering::byVDIM, int degree = 0);
   virtual ~RBFSpace();

   /// build the prolongation matrix with RBF
   void buildProlongationMatrix();

protected:
   /// mesh dimension
   int dim;
   /// number of radial basis function
   int numBasis;
   /// polynomial order
   int polyOrder;
   /// minimum number of basis for each element
   int req_nel;
   
   /// location of the basis centers
   mfem::Array<Vector *> basisCenter;
   /// the shape parameters
   //mfem::Array<DenseMatrix> shapeParam;
   /// selected basis for each element (currently it is fixed upon setup)
   mfem::Array<Array<int> *> selectedBasis;
   /// store the element centers
   mfem::Array<Vector *> elementCenter;
   /// array of map that holds the distance from element center to basisCenter
   // mfem::Array<std::map<int, double> *> elementBasisDist;
   mfem::Array<std::vector<double> *> elementBasisDist;

 
   /// Initialize the patches/stencil given poly order
   void InitializeStencil();
   /// Initialize the shape parameters
   void InitializeShapeParameter();
   std::vector<std::size_t> sort_indexes(const std::vector<double> &v);
};

} // end of namespace mfem
#endif