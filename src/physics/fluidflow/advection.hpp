// #ifndef MACH_ADVECTION
// #define MACH_ADVECTION

// #include "mfem.hpp"

// #include "solver.hpp"

// namespace mach
// {

// /// Linear advection integrator specialized to SBP operators
// class AdvectionIntegrator : public mfem::BilinearFormIntegrator
// {
// public:
//    /// Constructs a linear advection integrator.
//    /// \param[in] velc - represents the (possibly) spatially varying velocity
//    /// \param[in] alpha - scales the terms; can be used to move from lhs to rhs
//    AdvectionIntegrator(mfem::VectorCoefficient &velc, double a = 1.0)
//       : vel_coeff(velc) { alpha = a; }

//    /// Create the element stiffness matrix for linear advection.
//    /// \param[in] el - the finite element whose stiffness matrix we want
//    /// \param[in] Trans - defines the reference to physical element mapping
//    /// \param[out] elmat - the desired element stiffness matrix
//    virtual void AssembleElementMatrix(const mfem::FiniteElement &el,
//                                       mfem::ElementTransformation &Trans,
//                                       mfem::DenseMatrix &elmat);

// private:
// #ifndef MFEM_THREAD_SAFE
//    /// velocity in physical space
//    mfem::DenseMatrix vel;
//    /// scaled velocity in reference space
//    mfem::DenseMatrix velhat;
//    /// adjJ = |J|*dxi/dx = adj(dx/dxi)
//    mfem::DenseMatrix adjJ;
//    /// Storage for weak derivative operators
//    mfem::DenseMatrix Q;
//    /// reference to vel at a node
//    mfem::Vector vel_i;
//    /// reference to velhat at a node
//    mfem::Vector velhat_i;
//    /// reference to one component of velhat at all nodes
//    mfem::Vector Udi;
// #endif
//    /// represents the (possibly) spatially varying velocity field
//    mfem::VectorCoefficient &vel_coeff;
//    /// scales the terms; can be used to move to rhs/lhs
//    double alpha;
// };

// /// Local-projection stabilization integrator
// class AdvectLPSIntegrator : public mfem::BilinearFormIntegrator
// {
// public:
//    /// Constructs a local-projection stabilization integrator.
//    /// \param[in] velc - represents the (possibly) spatially varying velocity
//    /// \param[in] a - used to move from lhs to rhs
//    /// \param[in] diss_coeff - used to scale the magnitude of the LPS
//    AdvectLPSIntegrator(mfem::VectorCoefficient &velc, double a = 1.0,
//                  double diss_coeff = 1.0);

//    /// Create the stabilization matrix for the LPS operator.
//    /// \param[in] el - the finite element whose stabilization matrix we want
//    /// \param[in] Trans - defines the reference to physical element mapping
//    /// \param[out] elmat - the desired element stabilization matrix
//    virtual void AssembleElementMatrix(const mfem::FiniteElement &el,
//                                       mfem::ElementTransformation &Trans,
//                                       mfem::DenseMatrix &elmat);

// private:
// #ifndef MFEM_THREAD_SAFE
//    /// velocity in physical space
//    mfem::DenseMatrix vel;
//    /// adjJ = |J|*dxi/dx = adj(dx/dxi)
//    mfem::DenseMatrix adjJ;
//    /// stores the projection operator
//    mfem::DenseMatrix P;
//    /// scaled reference velocity at a point
//    mfem::Vector velhat_i;
//    /// scaling diagonal matrix, stored as a vector
//    mfem::Vector AH;
// #endif
//    /// represents the (possibly) spatially varying velocity field
//    mfem::VectorCoefficient &vel_coeff;
//    /// used to move to rhs/lhs
//    double alpha;
//    /// scales the magnitude of the LPS (merge with alpha?)
//    double lps_coeff;
// };

// /// Solver for linear advection problems
// template <int dim>
// class AdvectionSolver : public AbstractSolver
// {
// public:
//    /// Class constructor.
//    /// \param[in] opt_file_name - file where options are stored
//    /// \param[in] vel_field - function that defines the velocity field
//    AdvectionSolver(const std::string &opt_file_name,
//                    void (*vel_field)(const mfem::Vector &, mfem::Vector &));

// protected:
//    /// the velocity field
//    std::unique_ptr<mfem::VectorFunctionCoefficient> velocity;
//    /// the stiffness matrix bilinear form
//    std::unique_ptr<BilinearFormType> stiff;
//    /// stiffness matrix 
//    std::unique_ptr<MatrixType> stiff_matrix;

//    /// Add volume integrators to `res` based on `options`
//    /// \param[in] alpha - scales the data; used to move terms to rhs or lhs
//    virtual void addVolumeIntegrators(double alpha) {}

//    /// Add boundary-face integrators to `res` based on `options`
//    /// \param[in] alpha - scales the data; used to move terms to rhs or lhs
//    /// \returns num_bc - total number of BCs, including those of super classes
//    virtual void addBoundaryIntegrators(double alpha) {}

//    /// Add interior-face integrators to `res` based on `options`
//    /// \param[in] alpha - scales the data; used to move terms to rhs or lhs
//    virtual void addInterfaceIntegrators(double alpha) {}

//    /// Return the number of state variables
//    virtual int getNumState() { return 1; }
// };
    
// } // namespace mach

// #endif 
