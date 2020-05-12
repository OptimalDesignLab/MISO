#ifndef MFEM_GALER_DIFF
#define MFEM_GALER_DIFF

#include "mfem.hpp"
#include "solver.hpp"
#include "mach_types.hpp"

namespace mfem
{

/// Abstract class for Galerkin difference method using patch construction
class GalerkinDifference : public FiniteElementSpace
{

public:
   /// Class constructor.
   /// \param[in] opt_file_name - file where options are stored
   GalerkinDifference(const std::string &opt_file_name
                      = std::string("mach_options.json"));

   /// Another convenient constructor for test the prolongation matrix
   GalerkinDifference(int de, int vdim = 1,
                        int ordering = mfem::Ordering::byVDIM);
   
   GalerkinDifference(mfem::Mesh *pm, const mfem::FiniteElementCollection *f,
                      int vdim = 1, int ordering = mfem::Ordering::byVDIM,
                      int degree = 0);
   
   

   /// constructs the neighbour matrices for all mesh elements. 
   /// and second neighbours (shared vertices).
   /// \param[out] nmat1 - matrix of first neighbours
   /// \param[out] nmat1 - matrix of second neighbours
   /// \warning this function is going to be removed soon
   void BuildNeighbourMat(DenseMatrix &nmat1, DenseMatrix &nmat2);

   /// An overload function for build the densmatrix
   void BuildNeighbourMat(const Array<int> &els_id,
                          DenseMatrix &mat_cent,
                          DenseMatrix &mat_quad) const;

   /// constructs the neighbour set for given mesh element. 
   /// \param[in]  id - the id of the element for which we need neighbour
   /// \param[in]  req_n - the required number of neighbours for patch
   /// \param[out] nels - the set of neighbours (may contain more element than required)
   void GetNeighbourSet(int id, int req_n, Array<int> &nels) const;

   /// provides the center (barycenter) of an element
   /// \param[in]  id - the id of the element for which we need barycenter
   /// \param[out] cent - the vector of coordinates of center of an element
   void GetElementCenter(int id, mfem::Vector &cent) const;

   SparseMatrix *GetCP() { return cP; }

   /// Get the prolongation matrix in GD method
   virtual const Operator *GetProlongationMatrix() const
   { 
      if (!cP)
      {
         BuildGDProlongation();
         return cP;
      }
      else
      {
         return cP; 
      }
   }

   void checkpcp()
   {
      if (cP) {std::cout << "cP is set.\n";}
      mfem::SparseMatrix *P = dynamic_cast<mfem::SparseMatrix *> (cP);
      if (P) {std::cout << "convert succeeded.\n";}
   }

   /// Build the prolongation matrix in GD method
   void BuildGDProlongation() const;

   /// Assemble the local reconstruction matrix into the prolongation matrix
   /// \param[in] id - vector of element id in patch
   /// \param[in] local_mat - the local reconstruction matrix
   /// problem to be solved: how the ensure the oder of dofs consistent with other forms?
   void AssembleProlongationMatrix(const Array<int> &id,
                           const DenseMatrix &local_mat) const;

   /// check the duplication of quadrature points in the quad matrix
   bool duplicated(const mfem::Vector quad, const std::vector<double> data);

   virtual int GetTrueVSize() const {return nEle * vdim; }

protected:
   /// mesh dimension
   int dim;
   /// number of elements in mesh
   int nEle;
   /// degree of lagrange interpolation
   int degree;

   /// use pumi mesh
   //using MeshType = mfem::PumiMesh;
   /// object defining the computational mesh
   //std::unique_ptr<mach::MeshType> pmesh;
   /// the finite element collection pointer
   //std::unique_ptr<const mfem::FiniteElementCollection> fec;
   const mfem::FiniteElementCollection *fec; // not owned

};
} // end of namespace mach
#endif // end of GALERKIN DIFF
