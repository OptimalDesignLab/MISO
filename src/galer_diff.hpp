// #ifndef MFEM_GALER_DIFF
// #define MFEM_GALER_DIFF
// #include "mfem.hpp"
// #include "solver.hpp"
// #include "mach_types.hpp"
// #include "pumi.h"

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

//    /// Get the prolongation matrix in GD method
//    virtual const Operator *GetProlongationMatrix() const
//    { BuildGDProlongation(); return cP; }

//    /// Build the prolongation matrix in GD method
//    void BuildGDProlongation() const;

//    /// Assemble the local reconstruction matrix into the prolongation matrix
//    /// \param[in] local_mat - the local reconstruction matrix
//    /// problem to be solved: how the ensure the oder of dofs consistent with other forms?
//    void AssembleProlongationMatrix(const DenseMatrix local_mat) const;

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
// }
// #endif
