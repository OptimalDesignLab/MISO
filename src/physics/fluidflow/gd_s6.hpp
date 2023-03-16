#ifndef MFEM_GD
#define MFEM_GD
#include "mach_types.hpp"
#include "mfem.hpp"
#include "HYPRE.h"
using namespace mfem;
using namespace std;
namespace mfem
{
/// Abstract class for Galerkin difference method using patch construction
class ParGalerkinDifference : public ParFiniteElementSpace
{
public:
   /// Class constructor.
   /// \param[in] opt_file_name - file where options are stored

   ParGalerkinDifference(mach::MeshType *pm,
                         const mfem::FiniteElementCollection *f,
                         std::vector<bool> _embeddedElements,
                         int vdim = 1,
                         int ordering = mfem::Ordering::byVDIM,
                         int degree = 0,
                         MPI_Comm _comm = MPI_COMM_WORLD);

   /// constructs the neighbour matrices for all mesh elements.
   /// and second neighbours (shared vertices).
   void BuildNeighbourMat(const Array<int> &els_id,
                          DenseMatrix &mat_quad) const;

   /// constructs the neighbour set for given mesh element.
   /// \param[in]  id - the id of the element for which we need neighbour
   /// \param[in]  req_n - the required number of neighbours for patch
   /// \param[out] nels - the set of neighbours (may contain more element than
   /// required)
   void InitializeNeighbors(int id, int req_n, Array<int> &nels) const;

   /// @brief  sorts the neighbors based on distance from the element
   /// \param[in]  id - the id of the element for which we need stencil
   /// @param[in]  els_id - the neighbor ids
   /// @param[out] nels - the stencil elements sorted based on the distance
   void SortNeighbors(int id,
                      const Array<int> &els_id,
                      Array<int> &nels) const;

   /// constructs the stencil for given mesh element.
   /// \param[in]  id - the id of the element for which we need stencil
   /// \param[in]  req_n - the required number of elements for stencil
   /// \param[out] nels - the set of elements in the stencil
   void ConstructStencil(int id, int req_n, Array<int> &nels) const;

   /// provides the center (barycenter) of an element
   /// \param[in]  id - the id of the element for which we need barycenter
   /// \param[out] cent - the vector of coordinates of center of an element
   void GetElementCenter(int id, mfem::Vector &cent) const;

   double calcVandScale(int el_id, int dim, const DenseMatrix &x_center) const;
   void checkVandermondeCond(int dim,
                             int num_basis,
                             double &vandCond,
                             Array<int> &stencil_elid,
                             DenseMatrix &x_center,
                             DenseMatrix &V) const;
   void buildVandermondeMat(int dim,
                            int num_basis,
                            const Array<int> &els_id,
                            Array<int> &stencil_elid,
                            DenseMatrix &x_center,
                            DenseMatrix &V) const;
   void buildLSInterpolation(int elem_id,
                             int dim,
                             int degree,
                             const DenseMatrix &V,
                             const DenseMatrix &x_center,
                             const DenseMatrix &x_quad,
                             DenseMatrix &interp) const;
   SparseMatrix *GetCP() { return cP; }
   HypreParMatrix *GetP() { return P; }

   virtual HYPRE_Int GlobalVSize() const
   {
      return Dof_TrueDof_Matrix()->GetGlobalNumRows();
   }

   virtual HYPRE_Int GlobalTrueVSize() const
   {
      std::cout << "inside GlobalTrueVSize() " << std::endl;
      return Dof_TrueDof_Matrix()->GetGlobalNumCols();
   }

   void Build_Dof_TrueDof_Matrix();

   virtual HypreParMatrix *Dof_TrueDof_Matrix() const;

   /// Get the prolongation matrix in GD method
   virtual const Operator *GetProlongationMatrix() const
   {
      // if (!P)
      // {
      //    Build_Dof_TrueDof_Matrix();
      // }
      return P;
   }

   /// Get the R matrix which restricts a local dof vector to true dof vector.
   virtual const SparseMatrix *GetRestrictionMatrix() const
   {
      Dof_TrueDof_Matrix();
      return R;
   }

   void checkpcp()
   {
      if (cP)
      {
         std::cout << "cP is set.\n";
      }
   }

   /// Build the prolongation matrix in GD method
   void BuildGDProlongation();

   /// Assemble the local reconstruction matrix into the prolongation matrix
   /// \param[in] id - vector of element id in patch
   /// \param[in] local_mat - the local reconstruction matrix
   /// problem to be solved: how the ensure the oder of dofs consistent with
   /// other forms?
   void AssembleProlongationMatrix(const Array<int> &id,
                                   const DenseMatrix &local_mat) const;

   virtual int GetTrueVSize() const { return nEle * vdim; }

   // using ParFiniteElementSpace::GetTrueDofOffsets;
   HYPRE_Int *GetTrueDofOffsets() const { return tdof_offsets; }
   /** Create and return a new HypreParVector on the true dofs, which is
   owned by (i.e. it must be destroyed by) the calling function. */
   virtual HypreParVector *NewTrueDofVector()
   {
      HYPRE_Int vec_col_idx[2] = {0, GlobalTrueVSize()};
      return (new HypreParVector(comm, GlobalTrueVSize(), vec_col_idx));
   }

   // virtual void GetEssentialTrueDofs(const Array<int> &bdr_attr_is_ess,
   //                                   Array<int> &ess_tdof_list,
   //                                   int component = -1)
   // {
   //    cout << "problem in GetEssentialTrueDofs " << endl;
   //    ParFiniteElementSpace::GetEssentialTrueDofs(
   //        bdr_attr_is_ess, ess_tdof_list, component);
   //    cout << "no " << endl;
   // }
   /// function  that sort the element-basis distance
   std::vector<std::size_t> sort_indexes(const std::vector<double> &v) const;

private:
   /// Prolongation operator
   //    mutable HypreParMatrix *P;
   //    mutable SparseMatrix *R;
   HYPRE_IJMatrix ij_matrix;
   /// col and row partition arrays
   // mutable HYPRE_Int *mat_col_idx;
   // mutable HYPRE_Int *mat_row_idx;
   mutable HYPRE_Int *tdof_offsets;
   /// finite element collection
   const mfem::FiniteElementCollection *fec;  // not owned
   int col_start, col_end;
   /// the start and end colume index of each local prolongation operator
   int row_start, row_end;
   int el_offset;
   int pr;
   int gddofs;
   // Use the serial mesh to constructe prolongation matrix
   mfem::Mesh *full_mesh;
   // const mfem::FiniteElementSpace *full_fespace;
   /// total number of element
   int total_nel;
   HYPRE_Int local_tdof;
   HYPRE_Int total_tdof;

   // use HYPRE_IJMATRIX interface to construct the
   /// the actual prolongation matrix
   HYPRE_ParCSRMatrix prolong;

protected:
   /// mesh dimension
   int dim;
   /// number of elements in mesh
   int nEle;
   /// degree of lagrange interpolation
   int degree;
   /// communicator
   MPI_Comm comm;
   ///\Note: cut-cell stuff
   /// the vector of embedded elements
   std::vector<bool> embeddedElements;
};
}  // end of namespace mfem
#endif  // end of ParGALERKIN DIFF
