#ifndef MACH_MAGNETOSTATIC
#define MACH_MAGNETOSTATIC

#include "mfem.hpp"

#include "solver.hpp"
#include "coefficient.hpp"

namespace mach
{

/// Solver for magnetostatic electromagnetic problems
/// dim - number of spatial dimensions (only 3 supported)
class MagnetostaticSolver : public AbstractSolver
{
public:
	/// Class constructor.
   /// \param[in] opt_file_name - file where options are stored
   /// \param[in] smesh - if provided, defines the mesh for the problem
   /// \param[in] dim - number of dimensions
   /// \todo Can we infer dim some other way without using a template param?
   MagnetostaticSolver(const std::string &opt_file_name,
                       std::unique_ptr<mfem::Mesh> smesh = nullptr);

   /// Solve nonlinear magnetostatics problem using an MFEM Newton solver
   virtual void solveSteady();

private:
   /// Nedelec finite element collection
   std::unique_ptr<mfem::FiniteElementCollection> h_curl_coll;
   /// Raviart-Thomas finite element collection
   std::unique_ptr<mfem::FiniteElementCollection> h_div_coll;
   /// H1 finite element collection
   std::unique_ptr<mfem::FiniteElementCollection> h1_coll;
   ///L2 finite element collection
   std::unique_ptr<mfem::FiniteElementCollection> l2_coll;

   /// H(Curl) finite element space
   std::unique_ptr<SpaceType> h_curl_space;
   /// H(Div) finite element space
   std::unique_ptr<SpaceType> h_div_space;
   /// H1 finite element space
   std::unique_ptr<SpaceType> h1_space;
   /// L2 finite element space
   std::unique_ptr<SpaceType> l2_space;

   /// Magnetic vector potential A grid function
   std::unique_ptr<GridFunType> A;
   /// Magnetic flux density B = curl(A) grid function
   std::unique_ptr<GridFunType> B;
   /// Magnetic flux density B = curl(A) grid function in H(curl) space
   std::unique_ptr<GridFunType> B_dual;
   /// Magnetization grid function
   std::unique_ptr<GridFunType> M;
   /// Reluctivity grid function
   std::unique_ptr<GridFunType> nu_gf;

   /// TODO: delete? defined in abstract solver
   /// the spatial residual (a semilinear form)
   std::unique_ptr<NonlinearFormType> res;

   /// current source vector
   std::unique_ptr<GridFunType> current_vec;
   std::unique_ptr<GridFunType> div_free_current_vec;
   // /// current source vector with applied BC
   // std::unique_ptr<GridFunType> current_vec_BC;

   // std::unique_ptr<mfem::Coefficient> neg_one;

   /// mesh dependent reluctivity coefficient
   std::unique_ptr<MeshDependentCoefficient> nu;
   /// vector mesh dependent current density function coefficient
   std::unique_ptr<VectorMeshDependentCoefficient> current_coeff;
   /// vector mesh dependent magnetization coefficient
   std::unique_ptr<VectorMeshDependentCoefficient> mag_coeff;

   /// boundary condition marker array
   mfem::Array<int> ess_bdr;
   std::unique_ptr<mfem::VectorCoefficient> bc_coef;

   /// linear system solver used in Newton's method
   std::unique_ptr<mfem::HypreGMRES> solver;
   /// linear system preconditioner used in Newton's method
   std::unique_ptr<EMPrecType> prec;

   /// Newton solver
   mfem::NewtonSolver newton_solver;

   /// Material Library
   nlohmann::json materials;

   int dim;

   int getNumState() {return dim;};

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

   /// TODO: throw MachException if constructCurrent not called first
   ///       introdue some current constructed flag?
   /// assemble vector associated with current source
   /// \note - constructCurrent must be called before calling this
   void assembleCurrentSource();

   /// TODO: throw MachException if constructMagnetization or
   ///       assembleCurrentSource not called first
   /// \brief assemble magnetization source terms into rhs vector and add them
   ///        with the current source terms
   /// \note - constructMagnetization must be called before calling this
   void assembleMagnetizationSource(void);

   /// Function to compute seconday fields
   /// For magnetostatics, computes the magnetic flux density
   void computeSecondaryFields();

   /// function describing current density in phase A windings
   /// \param[in] x - position x in space of evaluation
   /// \param[out] J - current density at position x 
   static void phase_a_source(const mfem::Vector &x,
                              mfem::Vector &J);

   /// function describing current density in phase B windings
   /// \param[in] x - position x in space of evaluation
   /// \param[out] J - current density at position x 
   static void phase_b_source(const mfem::Vector &x,
                              mfem::Vector &J);

   /// function describing current density in phase C windings
   /// \param[in] x - position x in space of evaluation
   /// \param[out] J - current density at position x 
   static void phase_c_source(const mfem::Vector &x,
                              mfem::Vector &J);

   /// function describing permanent magnet magnetization pointing outwards
   /// \param[in] x - position x in space
   /// \param[out] M - magetic flux density at position x cause by permanent
   ///                 magnets
   static void magnetization_source_north(const mfem::Vector &x,
                                          mfem::Vector &M);
   
   /// function describing permanent magnet magnetization pointing inwards
   /// \param[in] x - position x in space
   /// \param[out] M - magetic flux density at position x cause by permanent
   ///                 magnets
   static void magnetization_source_south(const mfem::Vector &x,
                                          mfem::Vector &M);

   static void a_exact(const mfem::Vector &x, mfem::Vector &A);

   static void b_exact(const mfem::Vector &x, mfem::Vector &B);
};

} // namespace mach

#endif
