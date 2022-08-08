#ifndef MFEM_DGDSpace
#define MFEM_DGDSpace

#include "mfem.hpp"
#include "mach_types.hpp"
namespace mfem
{

// need to think a new name for this class
class DGDSpace : public mfem::FiniteElementSpace
{
public:
   /// class constructor
   DGDSpace(mfem::Mesh *m, const mfem::FiniteElementCollection *fec,
            mfem::Vector center, int degree, int extra, int vdim = 1,
            int ordering = mfem::Ordering::byVDIM, double c = 1e4);
   virtual ~DGDSpace();

   /// build the prolongation matrix
   //void buildProlongation() const;
   /// build the prolongation matrix based on the initialized stencil
   void buildProlongationMatrix(const mfem::Vector &bcenter);
   void addExtraBasis(int el_id);
   
   void buildDataMat(const int el_id, const mfem::Vector &basisCenter,
                     mfem::DenseMatrix &V, mfem::DenseMatrix &Vn);

   /// Solve and store the local prolongation coefficient
   /// \param[in] el_id - the element id
   /// \param[in] numDofs - degrees of freedom in the element
   /// \param[in] dof_coord - dofs coordinat location
   void solveLocalProlongationMat(const int el_id, const mfem::DenseMatrix &V,
                                     const mfem::DenseMatrix &Vn,
                                     mfem::DenseMatrix &localMat) const;

   /// build the element-wise polynomial basis matrix
   void buildElementPolyBasisMat(const int el_id,
                                 const mfem::Vector &basisCenter,
                                 const mfem::Array<mfem::Vector *> &dofs_coord,
                                 mfem::DenseMatrix &V,
                                 mfem::DenseMatrix &Vn) const;

   /// Assemble the local prolongation to the global matrix
   void AssembleProlongationMatrix(const int el_id, const mfem::DenseMatrix &localMat) const;

   /// compute the derivative of prolongation matrix w.r.t the ith basis center
   /// \param[in] i - the i th design parameter, either x or y coordinate
   /// \param[out] dpdc - derivative matrix
   void GetdPdc(const int i, const mfem::Vector &basisCenter, mfem::SparseMatrix &dpdc);

   void buildDerivDataMat(const int el_id, const int b_id, const int xyz,
                          const mfem::Vector &center,
                          mfem::DenseMatrix &V,
                          mfem::DenseMatrix &dV,
                          mfem::DenseMatrix &Vn,
                          mfem::DenseMatrix &dVn) const;
   
   void buildElementDerivMat(const int el_id, const int b_id,
                             const mfem::Vector &basisCenter,
                             const int xyz,
                             const mfem::Array<mfem::Vector *> &dofs_coord,
                             mfem::DenseMatrix &dV,
                             mfem::DenseMatrix &dVn) const;
   
   void AssembleDerivMatrix(const int el_id, const DenseMatrix &dpdc_block,
                            mfem::SparseMatrix &dpdc) const;
   mfem::Vector GetBasisCenter() { return basisCenterDummy; }
   void GetBasisCenter(const int b_id, mfem::Vector &center,
                       const mfem::Vector &basisCenter) const;
   /// some protected function
   void InitializeStencil(const mfem::Vector &basisCenter);
   virtual int GetTrueVSize() const {return vdim * numBasis;}
   inline int GetNDofs() const {return numBasis;}
   SparseMatrix *GetCP() { return cP; }
   virtual const Operator *GetProlongationMatrix() const { return cP; }
   void GetElementInfo(int el_id, mfem::Array<mfem::Vector *> &dofs_coord) const;
   double calcVandScale(const int el_id,
                        const mfem::Vector &el_center,
                        const mfem::Vector &basisCenter) const;
protected:
   /// mesh dimension
   int dim;
   /// number of radial basis function
   int numBasis;
   /// interpolatory polynomial order
   int interpOrder;
   /// number of required basis to constructe certain order polynomial
   int numReqBasis;
   int extra; // dummy variable, not used
   double cond;
   bool adjustCondition;

   /// indicator of whether using extra basis
   std::vector<std::vector<int>> selectedBasis;
   std::vector<std::vector<int>> selectedElement;
   std::vector<std::vector<double>> elementBasisDist;
   std::vector<std::vector<size_t>> sortedEBDistRank;
   mutable std::vector<int> extraCenter;
   /// the actual polynomial order of on each element
   
   /// location of the basis centers
   mfem::Vector basisCenterDummy;
   /// selected basis for each element (currently it is fixed upon setup)
   // mfem::Array<mfem::Array<int> *> selectedBasis;
   // mfem::Array<mfem::Array<int> *> selectedElement;
   // mfem::Array<std::vector<double> *> elementBasisDist;
   // mfem::Array<std::vector<size_t> *> sortedEBDistRank;

   /// local element prolongation matrix coefficient
   mutable mfem::Array<mfem::DenseMatrix *> coef;
   /// function  that sort the element-basis distance
   std::vector<std::size_t> sort_indexes(const std::vector<double> &v);
};

} // end of namespace mfem
#endif