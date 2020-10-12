#ifndef MACH_MAGNETOSTATIC
#define MACH_MAGNETOSTATIC

#include <mpi.h>

#include "mfem.hpp"
#include "json.hpp"

#include "solver.hpp"
#include "coefficient.hpp"

namespace mach
{

class MeshMovementSolver;

/// Solver for magnetostatic electromagnetic problems
/// dim - number of spatial dimensions (only 3 supported)
class MagnetostaticSolver : public AbstractSolver
{
public:
   /// Class constructor.
   /// \param[in] opt_file_name - file where options are stored
   /// \param[in] smesh - if provided, defines the mesh for the problem
   /// \param[in] comm - MPI communicator for parallel operations
   MagnetostaticSolver(const nlohmann::json &opt_file_name,
                       std::unique_ptr<mfem::Mesh> smesh,
                       MPI_Comm comm);

   /// Class constructor.
   /// \param[in] options - pre-loaded JSON options object
   /// \param[in] smesh - if provided, defines the mesh for the problem
   MagnetostaticSolver(const nlohmann::json &options,
                       std::unique_ptr<mfem::Mesh> smesh);

   ~MagnetostaticSolver();

   /// Write the mesh and solution to a vtk file
   /// \param[in] file_name - prefix file name **without** .vtk extension
   /// \param[in] refine - if >=0, indicates the number of refinements to make
   /// \note the `refine` argument is useful for high-order meshes and
   /// solutions; it divides the elements up so it is possible to visualize.
   void printSolution(const std::string &file_name, int refine = -1) override;

   /// \brief Returns a vector of pointers to grid functions that define fields
   /// returns {A, B}
   std::vector<GridFunType*> getFields() override;

   /// TODO: have this accept a string input chosing the functional
   /// Compute the sensitivity of the functional to the mesh volume
   /// nodes, using appropriate mesh sensitivity integrators. This function will
   /// compute the adjoint.
   mfem::GridFunction* getMeshSensitivities() override;

   /// perturb the whole mesh and finite difference
   void verifyMeshSensitivities();

   void Update() override;

private:
   // /// Nedelec finite element collection
   // std::unique_ptr<mfem::FiniteElementCollection> h_curl_coll;
   /// Raviart-Thomas finite element collection
   std::unique_ptr<mfem::FiniteElementCollection> h_div_coll;
   /// H1 finite element collection
   std::unique_ptr<mfem::FiniteElementCollection> h1_coll;
   ///L2 finite element collection
   std::unique_ptr<mfem::FiniteElementCollection> l2_coll;

   // /// H(Curl) finite element space
   // std::unique_ptr<SpaceType> h_curl_space;
   /// H(Div) finite element space
   std::unique_ptr<SpaceType> h_div_space;
   /// H1 finite element space
   std::unique_ptr<SpaceType> h1_space;
   /// L2 finite element space
   std::unique_ptr<SpaceType> l2_space;

   // /// Magnetic vector potential A grid function
   // std::unique_ptr<GridFunType> A;
   /// Magnetic flux density B = curl(A) grid function
   std::unique_ptr<GridFunType> B;
   /// Magnetic flux density B = curl(A) grid function in H(curl) space
   std::unique_ptr<GridFunType> B_dual;
   /// Magnetization grid function
   std::unique_ptr<GridFunType> M;

   // /// TODO: delete? defined in abstract solver
   // /// the spatial residual (a semilinear form)
   // std::unique_ptr<NonlinearFormType> res;

   /// current source vector
   // std::unique_ptr<GridFunType> current_vec;
   std::unique_ptr<GridFunType> div_free_current_vec;

   /// mesh dependent reluctivity coefficient
   std::unique_ptr<MeshDependentCoefficient> nu;
   /// vector mesh dependent current density function coefficient
   std::unique_ptr<VectorMeshDependentCoefficient> current_coeff;
   /// vector mesh dependent magnetization coefficient
   std::unique_ptr<VectorMeshDependentCoefficient> mag_coeff;

   /// boundary condition marker array
   // mfem::Array<int> ess_bdr;
   std::unique_ptr<mfem::VectorCoefficient> bc_coef;

   int dim;
   void constructForms() override;

   /// Construct various coefficients
   void constructCoefficients() override;

   void addMassIntegrators(double alpha) override;
   void addResVolumeIntegrators(double alpha) override;
   void assembleLoadVector(double alpha) override;

   /// mark which boundaries are essential
   void setEssentialBoundaries() override;

   int getNumState() override {return 1;};

   void initialHook(const mfem::ParGridFunction &state) override;

   // void iterationHook(int iter, double t, double dt,
   //                    const mfem::ParGridFunction &state) override;

   // bool iterationExit(int iter, double t, double t_final, double dt,
   //                    const mfem::ParGridFunction &state) override;
   
   void terminalHook(int iter, double t_final,
                     const mfem::ParGridFunction &state) override;

   /// Create `output` based on `options` and add approporiate integrators
   void addOutputs() override;

   /// Solve nonlinear magnetostatics problem using an MFEM Newton solver
   // void solveUnsteady(mfem::ParGridFunction &state) override;

   /// static member variables used inside static member functions
   /// magnetization_source and winding_current_source
   /// values set by options in setStaticMembers
   static double remnant_flux;
   static double mag_mu_r;
   static double fill_factor;
   static double current_density;

   /// set the values of static member variables used based on options file
   void setStaticMembers();

   /// construct mesh dependent coefficient for reluctivity
   /// \param[in] alpha - used to move to lhs or rhs
   void constructReluctivity();

   /// construct mesh dependent coefficient for magnetization
   /// \param[in] alpha - used to move to lhs or rhs
   void constructMagnetization();

   /// construct vector mesh dependent coefficient for current source
   /// \param[in] alpha - used to move to lhs or rhs
   void constructCurrent();

public:
   /// TODO: throw MachException if constructCurrent not called first
   ///       introdue some current constructed flag?
   /// assemble vector associated with current source
   /// \note - constructCurrent must be called before calling this
   void assembleCurrentSource();

   /// Assemble mesh sensitivities of current source vector
   /// \param[in] psi_a - Adjoint vector
   /// \param[out] mesh_sens - mesh sensitivitites
   /// \note this method will not initialize mesh_sens, but will add to it
   void getCurrentSourceMeshSens(const mfem::GridFunction &psi_a,
                                 mfem::Vector &mesh_sens);

   std::unique_ptr<GridFunType> residual;
   /// return the residual as a vector
   mfem::Vector* getResidual();

   /// Get the derivative of the residual with respect to the current density
   mfem::Vector* getResidualCurrentDensitySensitivity();

   /// Get the total derivative of a functional with respect to the current
   /// density
   /// \param[in] fun - which functional to get sensitivity with respect to
   double getFunctionalCurrentDensitySensitivity(const std::string &fun);

private:
   /// TODO: throw MachException if constructMagnetization or
   ///       assembleCurrentSource not called first
   /// \brief assemble magnetization source terms into rhs vector and add them
   ///        with the current source terms
   /// \note - constructMagnetization must be called before calling this
   void assembleMagnetizationSource();

   /// Function to compute seconday fields
   /// For magnetostatics, computes the magnetic flux density
   void computeSecondaryFields(const mfem::ParGridFunction &state);

   /// function describing current density in phase A windings
   /// \param[in] x - position x in space of evaluation
   /// \param[out] J - current density at position x 
   static void phaseACurrentSource(const mfem::Vector &x,
                                   mfem::Vector &J);

   static void phaseACurrentSourceRevDiff(const mfem::Vector &x,
                                          const mfem::Vector &V_bar,
                                          mfem::Vector &x_bar);

   /// function describing current density in phase B windings
   /// \param[in] x - position x in space of evaluation
   /// \param[out] J - current density at position x 
   static void phaseBCurrentSource(const mfem::Vector &x,
                                   mfem::Vector &J);

   static void phaseBCurrentSourceRevDiff(const mfem::Vector &x,
                                          const mfem::Vector &V_bar,
                                          mfem::Vector &x_bar);

   /// function describing current density in phase C windings
   /// \param[in] x - position x in space of evaluation
   /// \param[out] J - current density at position x 
   static void phaseCCurrentSource(const mfem::Vector &x,
                                   mfem::Vector &J);

   static void phaseCCurrentSourceRevDiff(const mfem::Vector &x,
                                          const mfem::Vector &V_bar,
                                          mfem::Vector &x_bar);

   /// function describing permanent magnet magnetization pointing outwards
   /// \param[in] x - position x in space
   /// \param[out] M - magetic flux density at position x cause by permanent
   ///                 magnets
   static void northMagnetizationSource(const mfem::Vector &x,
                                        mfem::Vector &M);

   /// \param[in] x - position x in space of evaluation
   /// \param[in] V_bar - 
   /// \param[out] x_bar - V_bar^T Jacobian
   static void northMagnetizationSourceRevDiff(const mfem::Vector &x,
                                               const mfem::Vector &V_bar,
                                               mfem::Vector &x_bar);

   /// function describing permanent magnet magnetization pointing inwards
   /// \param[in] x - position x in space
   /// \param[out] M - magetic flux density at position x cause by permanent
   ///                 magnets
   static void southMagnetizationSource(const mfem::Vector &x,
                                        mfem::Vector &M);

   /// \param[in] x - position x in space of evaluation
   /// \param[in] V_bar - 
   /// \param[out] x_bar - V_bar^T Jacobian
   static void southMagnetizationSourceRevDiff(const mfem::Vector &x,
                                               const mfem::Vector &V_bar,
                                               mfem::Vector &x_bar);


   /// function defining current density aligned with the x axis
   /// \param[in] x - position x in space of evaluation
   /// \param[out] J - current density at position x 
   static void xAxisCurrentSource(const mfem::Vector &x,
                                  mfem::Vector &J);

   static void xAxisCurrentSourceRevDiff(const mfem::Vector &x,
                                         const mfem::Vector &V_bar,
                                         mfem::Vector &x_bar);
   
   /// function defining current density aligned with the x axis
   /// \param[in] x - position x in space of evaluation
   /// \param[out] J - current density at position x 
   static void yAxisCurrentSource(const mfem::Vector &x,
                                  mfem::Vector &J);

   static void yAxisCurrentSourceRevDiff(const mfem::Vector &x,
                                         const mfem::Vector &V_bar,
                                         mfem::Vector &x_bar);

   /// function defining current density aligned with the x axis
   /// \param[in] x - position x in space of evaluation
   /// \param[out] J - current density at position x 
   static void zAxisCurrentSource(const mfem::Vector &x,
                                  mfem::Vector &J);

   static void zAxisCurrentSourceRevDiff(const mfem::Vector &x,
                                         const mfem::Vector &V_bar,
                                         mfem::Vector &x_bar);

   /// function defining current density aligned in a ring around the z axis
   /// \param[in] x - position x in space of evaluation
   /// \param[out] J - current density at position x 
   static void ringCurrentSource(const mfem::Vector &x,
                                 mfem::Vector &J);

   static void ringCurrentSourceRevDiff(const mfem::Vector &x,
                                        const mfem::Vector &V_bar,
                                        mfem::Vector &x_bar);

   /// function defining current density for simple box problem
   /// \param[in] x - position x in space of evaluation
   /// \param[out] J - current density at position x 
   static void box1CurrentSource(const mfem::Vector &x,
                                   mfem::Vector &J);

   static void box1CurrentSourceRevDiff(const mfem::Vector &x,
                                        const mfem::Vector &V_bar,
                                        mfem::Vector &x_bar);

   /// function defining current density for simple box problem
   /// \param[in] x - position x in space of evaluation
   /// \param[out] J - current density at position x 
   static void box2CurrentSource(const mfem::Vector &x,
                                 mfem::Vector &J);

   static void box2CurrentSourceRevDiff(const mfem::Vector &x,
                                        const mfem::Vector &V_bar,
                                        mfem::Vector &x_bar);

   /// function defining magnetization aligned with the x axis
   /// \param[in] x - position x in space of evaluation
   /// \param[out] J - current density at position x
   static void xAxisMagnetizationSource(const mfem::Vector &x,
                                        mfem::Vector &M);

   /// function defining magnetization aligned with the x axis
   /// \param[in] x - position x in space of evaluation
   /// \param[in] V_bar - 
   /// \param[out] x_bar - V_bar^T Jacobian
   static void xAxisMagnetizationSourceRevDiff(const mfem::Vector &x,
                                               const mfem::Vector &V_bar,
                                               mfem::Vector &x_bar);

   /// function defining magnetization aligned with the y axis
   /// \param[in] x - position x in space of evaluation
   /// \param[out] J - current density at position x
   static void yAxisMagnetizationSource(const mfem::Vector &x,
                                        mfem::Vector &M);

   /// function defining magnetization aligned with the x axis
   /// \param[in] x - position x in space of evaluation
   /// \param[in] V_bar - 
   /// \param[out] x_bar - V_bar^T Jacobian
   static void yAxisMagnetizationSourceRevDiff(const mfem::Vector &x,
                                               const mfem::Vector &V_bar,
                                               mfem::Vector &x_bar);

   /// function defining magnetization aligned with the z axis
   /// \param[in] x - position x in space of evaluation
   /// \param[out] J - current density at position x
   static void zAxisMagnetizationSource(const mfem::Vector &x,
                                        mfem::Vector &M);

   /// function defining magnetization aligned with the x axis
   /// \param[in] x - position x in space of evaluation
   /// \param[in] V_bar - 
   /// \param[out] x_bar - V_bar^T Jacobian
   static void zAxisMagnetizationSourceRevDiff(const mfem::Vector &x,
                                               const mfem::Vector &V_bar,
                                               mfem::Vector &x_bar);

   static void a_exact(const mfem::Vector &x, mfem::Vector &A);

   static void b_exact(const mfem::Vector &x, mfem::Vector &B);

   friend SolverPtr createSolver<MagnetostaticSolver>(
       const nlohmann::json &opt_file_name,
       std::unique_ptr<mfem::Mesh> smesh,
       MPI_Comm comm);
};

} // namespace mach

#endif
