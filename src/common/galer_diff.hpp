#ifndef MFEM_GALER_DIFF
#define MFEM_GALER_DIFF
#include "mfem.hpp"
#include "HYPRE.h"

#include "mach_types.hpp"


namespace mfem
{

class ParGDSpace : public ParFiniteElementSpace
{
public:	
	/// class constructor
	/// \param[in] pm - the mfem parallel mesh
	/// \param[in] global_fes - serial finite element space
	/// \param[in] partitioning - user generated mesh partitioning method
	/// \param[in] fec - finite element collection
	/// \param[in] vdim - number of variable per degree of freedom
	/// \param[in] ordering - method for orderding the dofs, usually is byVDIM
	/// \param[in] de - prolongation operator degree
	/// \param[in] p - a temporal variable used for print some result
	ParGDSpace(mfem::Mesh *m, mfem::ParMesh *pm, const mfem::FiniteElementSpace *global_fes,
				  const int *partitioning, const mfem::FiniteElementCollection *fec,
				  int vdim = 1, int ordering = mfem::Ordering::byVDIM, int de = 1, int p = 0);
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
	void BuildNeighbourMat(const mfem::Array<int> &els_id, mfem::DenseMatrix &mat_cent,
						 		  mfem::DenseMatrix &mat_quad) const;

	/// Get the element bary center coordinate
	/// \param[in] id - element id
	/// \param[in,out] cent - element center coordinate
	void GetElementCenter(int id, mfem::Vector &cent) const;

	/// Assemble the local prolongation matrix to parallel hypreIJMatrix
	/// \param[in] els_id - element ids in patch
	/// \param[in] local_mat - local prolongation matrix
	void AssembleProlongationMatrix(const mfem::Array<int> &els_id,
											  const mfem::DenseMatrix &local_mat) const;
   
   /// return the number of dofs in GD space, which is the number of element
   int GetNDofs() {return gddofs;}

   /// return the number of true dofs 
   int GetTrueVSize() {return vdim * gddofs;}

   // HypreParVector *NewTrueDofVector()
   // {
	//    if (GetMyRank() == pr)
	//    {
	// 		std::cout << "ParGDSpace::NewTrueDofVector() is called.\n";
	//    }
	// 	HYPRE_BigInt col_starts[2];
	// 	col_starts[0] = GetVDim() * el_offset;
	// 	col_starts[1] = GetVDim() * (el_offset+GetParMesh()->GetNE());
	// 	if (GetMyRank() == pr )
	// 	{
	// 		std::cout << "GlobalTrueVSize is " << GetVDim()*total_nel << ". ";
	// 		std::cout << "True dof offset is " <<  col_starts[0]
	// 					<< ' ' << col_starts[1] << '\n';
	// 	}

   //    return (new HypreParVector(GetComm(), GetVDim()*total_nel, col_starts));
   // }

private:
	/// mesh dimenstion
	int dim;
   /// degree of the prolongation operator
   int degree;
	/// the start and end row index of each local prolongation operator

   int gddofs;
	int col_start, col_end;
	/// the start and end colume index of each local prolongation operator
	int row_start, row_end;
	int el_offset;
	int pr;

	HypreParMatrix *ptranspose;

	// Use the serial mesh to constructe prolongation matrix 
	mfem::Mesh *full_mesh;
	const mfem::FiniteElementSpace *full_fespace;
	/// total number of element
	int total_nel;
	HYPRE_Int local_tdof;
	HYPRE_Int total_tdof;

	// use HYPRE_IJMATRIX interface to construct the 
	/// the actual prolongation matrix
	HYPRE_ParCSRMatrix prolong;
	HYPRE_IJMatrix ij_matrix;
};

} // end of namespace

#endif