#ifndef MFEM_GD
#define MFEM_GD
#include "mach_types.hpp"
#include "mfem.hpp"
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
                         int vdim = 1,
                         int ordering = mfem::Ordering::byVDIM,
                         int degree = 0,
                         MPI_Comm _comm = MPI_COMM_WORLD);

   /// constructs the neighbour matrices for all mesh elements.
   /// and second neighbours (shared vertices).
   void BuildNeighbourMat(const Array<int> &els_id,
                          DenseMatrix &mat_cent,
                          DenseMatrix &mat_quad) const;

   /// constructs the neighbour set for given mesh element.
   /// \param[in]  id - the id of the element for which we need neighbour
   /// \param[in]  req_n - the required number of neighbours for patch
   /// \param[out] nels - the set of neighbours (may contain more element than
   /// required)
   void GetNeighbourSet(int id, int req_n, Array<int> &nels) const;

   /// provides the center (barycenter) of an element
   /// \param[in]  id - the id of the element for which we need barycenter
   /// \param[out] cent - the vector of coordinates of center of an element
   void GetElementCenter(int id, mfem::Vector &cent) const;

   SparseMatrix *GetCP() { return cP; }
   HypreParMatrix *GetP() { return P; }

   virtual HYPRE_Int GlobalVSize() const
   {
      return Dof_TrueDof_Matrix()->GetGlobalNumRows();
   }

   virtual HYPRE_Int GlobalTrueVSize() const
   {
      return Dof_TrueDof_Matrix()->GetGlobalNumCols();
   }

   void Build_Dof_TrueDof_Matrix() const;

   virtual HypreParMatrix *Dof_TrueDof_Matrix() const;

   /// Get the prolongation matrix in GD method
   virtual const Operator *GetProlongationMatrix() const
   {
      if (!P)
      {
         Build_Dof_TrueDof_Matrix();
      }
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
   void BuildGDProlongation() const;

   /// Assemble the local reconstruction matrix into the prolongation matrix
   /// \param[in] id - vector of element id in patch
   /// \param[in] local_mat - the local reconstruction matrix
   /// problem to be solved: how the ensure the oder of dofs consistent with
   /// other forms?
   void AssembleProlongationMatrix(const Array<int> &id,
                                   const DenseMatrix &local_mat) const;

   virtual int GetTrueVSize() const { return nEle * vdim; }

   using ParFiniteElementSpace::GetTrueDofOffsets;

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

private:
   /// Prolongation operator
//    mutable HypreParMatrix *P;
//    mutable SparseMatrix *R;

protected:
   /// mesh dimension
   int dim;
   /// number of elements in mesh
   int nEle;
   /// degree of lagrange interpolation
   int degree;
   /// communicator
   MPI_Comm comm;
   /// col and row partition arrays
   mutable HYPRE_Int *mat_col_idx;
   mutable HYPRE_Int *mat_row_idx;
   /// finite element collection
   const mfem::FiniteElementCollection *fec;  // not owned
};
}  // end of namespace mfem
#endif  // end of ParGALERKIN DIFF
