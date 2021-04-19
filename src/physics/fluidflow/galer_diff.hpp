#ifndef MFEM_GALER_DIFF
#define MFEM_GALER_DIFF
#include "mfem.hpp"
#include "solver.hpp"
#include "mach_types.hpp"
#include "pumi.h"
#include "apfMDS.h"
#include "HYPRE.h"
#include <iostream>
namespace mfem
{

class ParGDSpace : public ParFiniteElementSpace
{
public:
	/// Class constructor
	/// \param[in] apf_mesh - the apf mesh object that
	/// \param[in] parmesh - the parallel mesh created from apf mesh
	/// \param[in] f - the finite element space
	/// \param[in] vdim - number of state variables
	/// \param[in] ordering - method for ordering the dofs, usually is byVDIM
	/// \param[in] de - prolongation operator degree
	ParGDSpace(apf::Mesh2 *apf_mesh, mfem::ParMesh *pm, const mfem::FiniteElementCollection *f,
				  int vdim = 1, int ordiering = mfem::Ordering::byVDIM, int de = 1, int p = 0);

	/// Construct the parallel prolongation operator
	void BuildProlongationOperator();

	/// Constructs the patch for given element id
	/// \param[in] id - element id
	/// \param[in] req_n - number of required elemment in patch
	/// \param[in,out] els_id - elements id(s) in patch for id th element
	void GetNeighbourSet(int id, int req_n, mfem::Array<int> &els_id);

	/// Constructs the matrices for element quadrature points and centers
	/// \param[in] els_id - element ids in patch
	/// \param[in,out] mat_cent - matrix holding element centers coordinates
	/// \param[in,out] mat_quad - matrix holding quadrature poinst coordinates
	void GetNeighbourMat(mfem::Array<int> &els_id, mfem::DenseMatrix &mat_cent,
						 mfem::DenseMatrix &mat_quad) const;

	/// Get the element bary center coordinate
	/// \param[in] id - element id
	/// \param[in,out] cent - element center coordinate
	void GetElementCenter(int id, mfem::Vector &cent) const;

	/// Assemble the local prolongation matrix to parallel hypreIJMatrix
	/// \param[in] els_id - element ids in patch
	/// \param[in] local_mat - local prolongation matrix
	void AssembleProlongationMatrix(const mfem::Array<int> &els_id,
									mfem::DenseMatrix &local_mat);

	// HYPRE_Int GlobalTrueVSize() const
	// { return total_tdof;}

   HypreParVector *NewTrueDofVector()
   {
      std::cout << "ParGDSpace::NewTrueDofVector is called.\n";
		Array<HYPRE_Int> fake_dofs(2);
		fake_dofs[0] = GetParMesh()->GetElementOffset();
		fake_dofs[1] = fake_dofs[0] + local_tdof;
      std::cout << "GlobalTrueVSize is " << GlobalTrueVSize() << ". ";
      std::cout << "True dof offset is " <<  fake_dofs[0]
                << ' ' << fake_dofs[1] << '\n';
      return (new HypreParVector(GetComm(), GlobalTrueVSize(), fake_dofs.GetData()));
    }
private:
   /// the pumi mesh object
   apf::Mesh2 *pumi_mesh; 

	/// mesh dimenstion
	int dim;
   /// degree of the prolongation operator
   int degree;
	/// the start and end row index of each local prolongation operator
	/// what is the index exceed the limit?
	int dof_start, dof_end;
	/// the start and end colume index of each local prolongation operator
	int el_start, el_end;
	int pr;

	/// total number of element
	HYPRE_Int local_tdof;
	HYPRE_Int total_tdof;

	// use HYPRE_IJMATRIX interface to construct the 
	/// the actual prolongation matrix
	HYPRE_ParCSRMatrix prolong;
	HYPRE_IJMatrix ij_matrix;
};

} // end of namespace

#endif 

// namespace mfem
// {

// /// Abstract class for Galerkin difference method using patch construction
// class GalerkinDifference : public FiniteElementSpace
// {

// public:
//    /// Class constructor.
//    /// \param[in] opt_file_name - file where options are stored
//    GalerkinDifference(const std::string &opt_file_name =
//                         std::string("mach_options.json"));

//    /// constructs the neighbour matrices for all mesh elements. 
//    /// and second neighbours (shared vertices).
//    /// \param[out] nmat1 - matrix of first neighbours
//    /// \param[out] nmat1 - matrix of second neighbours
//    /// \warning this function is going to be removed soon
//    void BuildNeighbourMat(DenseMatrix &nmat1, DenseMatrix &nmat2);

//    /// An overload function for build the densmatrix
//    void BuildNeighbourMat(const std::vector<int> els_id,
//                           DenseMatrix &mat_cent,
//                           DenseMatrix &mat_quad);

//    /// constructs the neighbour set for given mesh element. 
//    /// \param[in]  id - the id of the element for which we need neighbour
//    /// \param[in]  req_n - the required number of neighbours for patch
//    /// \param[out] nels - the set of neighbours (may contain more element than required)
//    void GetNeighbourSet(int id, int req_n, std::vector<int> &nels);

//    /// provides the center (barycenter) of an element
//    /// \param[in]  id - the id of the element for which we need barycenter
//    /// \param[out] cent - the vector of coordinates of center of an element
//    void GetElementCenter(int id, mfem::Vector &cent);

//    void GetEelementQuad(int id, mfem::)

//    /// Get the prolongation matrix in GD method
//    virtual const Operator *GetProlongationMatrix() const
//    { BuildGDProlongation(); return cP; }

//    /// Build the prolongation matrix in GD method
//    void BuildGDProlongation() const;

//    /// Assemble the local reconstruction matrix into the prolongation matrix
//    /// \param[in] id - vector of element id in patch
//    /// \param[in] local_mat - the local reconstruction matrix
//    /// problem to be solved: how the ensure the oder of dofs consistent with other forms?
//    void AssembleProlongationMatrix(const std::vector<int> id,
//                            const DenseMatrix local_mat) const;

//    /// check the duplication of quadrature points in the quad matrix
//    bool duplicated(const mfem::Vector quad, const std::vector<double> data);

// protected:
//    /// mesh dimension
//    int dim;
//    /// number of elements in mesh
//    int nEle;
//    /// degree of lagrange interpolation
//    int degree;   
//    /// use pumi mesh
//    using MeshType = mfem::PumiMesh;
//    /// object defining the computational mesh
//    std::unique_ptr<MeshType> mesh;

// #ifdef MFEM_USE_MPI
//    /// communicator used by MPI group for communication
//    MPI_Comm comm;
// #ifdef MFEM_USE_PUMI
//    /// create pumi mesh object
//    apf::Mesh2* pumi_mesh;
// #endif
// #endif

// };

// } // end of namespace mach
// #endif
