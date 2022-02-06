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
            mfem::Array<Vector*> center, double shape, int vdim = 1, int extra = 1,
            int ordering = mfem::Ordering::byVDIM, int degree = 0);
   virtual ~RBFSpace();

   /// build the prolongation matrix with RBF
   void buildProlongationMatrix();

   /// build the dof coordinate matrix
   /// note: Assume the mesh only has one type of element
   /// \param[in] el_id - global element id
   /// \param[in/out dofs - matrix that hold the dofs' coordinates
   void buildDofMat(int el_id, const int num_dofs, 
                    const mfem::FiniteElement *fe,
                    mfem::Array<mfem::Vector *> &dofs) const;
   
   void buildDataMat(const int el_id, mfem::DenseMatrix &W, mfem::DenseMatrix &V,
                     mfem::DenseMatrix &Wn, mfem::DenseMatrix &Vn,
                     mfem::DenseMatrix &WV, mfem::DenseMatrix &WnVn);

   /// Solve and store the local prolongation coefficient
   /// \param[in] el_id - the element id
   /// \param[in] numDofs - degrees of freedom in the element
   /// \param[in] dof_coord - dofs coordinat location
   void solveLocalProlongationMat(const int el_id, const mfem::DenseMatrix &WV,
                                     const mfem::DenseMatrix &WnVn,
                                     mfem::DenseMatrix &localMat);

   // /// Assemble the global prolongation matrix
   // void AssembleProlongationMatrix() const;

   /// build the element-wise radial basis matrix
   void buildElementRadialBasisMat(const int el_id,
                                   const int numDofs,
                                   const mfem::Array<mfem::Vector *> &dofs_coord,
                                   mfem::DenseMatrix &W,
                                   mfem::DenseMatrix &Wn);

   /// build the element-wise polynomial basis matrix
   void buildElementPolyBasisMat(const int el_id, const int numPolyBasis,
                                 const int numDofs,
                                 const mfem::Array<mfem::Vector *> &dofs_coord,
                                 mfem::DenseMatrix &V,
                                 mfem::DenseMatrix &Vn);
   
   /// build the j
   void buildWVMat(const mfem::DenseMatrix &W, const mfem::DenseMatrix &V,
                   mfem::DenseMatrix &WV);
   
   /// build the WnVn matrix
   void buildWnVnMat(const mfem::DenseMatrix &Wn, const mfem::DenseMatrix &Vn,
                     mfem::DenseMatrix &WnVn);

   /// Assemble the local prolongation to the global matrix
   void AssembleProlongationMatrix(const int el_id, const mfem::DenseMatrix &localMat);

protected:
   /// mesh dimension
   int dim;
   /// number of radial basis function
   int numBasis;
   /// number of polynomial basis
   int numPolyBasis;
   /// polynomial order
   int polyOrder;
   /// minimum number of basis for each element
   int req_basis;
   /// number of extra basis included in an element stencil
   int extra_basis;
   
   /// location of the basis centers
   mfem::Array<mfem::Vector *> basisCenter;
   /// the shape parameters
   mfem::DenseMatrix shapeParam;
   /// selected basis for each element (currently it is fixed upon setup)
   mfem::Array<mfem::Array<int> *> selectedBasis;
   mfem::Array<mfem::Array<int> *> selectedElement;
   /// store the element centers
   mfem::Array<mfem::Vector *> elementCenter;
   /// array of map that holds the distance from element center to basisCenter
   // mfem::Array<std::map<int, double> *> elementBasisDist;
   mfem::Array<std::vector<double> *> elementBasisDist;
   // local element prolongation matrix coefficient
   mfem::Array<mfem::DenseMatrix *> coef;
   /// Initialize the patches/stencil given poly order
   void InitializeStencil();
   /// Initialize the shape parameters
   void InitializeShapeParameter();
   std::vector<std::size_t> sort_indexes(const std::vector<double> &v);
};

} // end of namespace mfem
#endif