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

   /// Solve and store the local prolongation coefficient
   /// \param[in] el_id - the element id
   /// \param[in] numDofs - degrees of freedom in the element
   /// \param[in] dof_coord - dofs coordinat location
   void solveProlongationCoefficient(const int el_id, const int numDofs,
                                     const mfem::Array<mfem::Vector *> &dof_coord);

   // /// Assemble the global prolongation matrix
   // void AssembleProlongationMatrix() const;

   /// build the element element-wise basis matrix
   void buildElementRadialBasisMat(const int el_id,
                                   const int numDofs,
                                   const mfem::Array<mfem::Vector *> &dofs_coord,
                                   mfem::DenseMatrix &W,
                                   mfem::DenseMatrix &Wn);

   void buildElementPolyBasisMat(const int el_id, const int numPolyBasis,
                                 const int numDofs,
                                 const mfem::Array<mfem::Vector *> &dofs_coord,
                                 mfem::DenseMatrix &V,
                                 mfem::DenseMatrix &Vn);

protected:
   /// mesh dimension
   int dim;
   /// number of radial basis function
   int numBasis;
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