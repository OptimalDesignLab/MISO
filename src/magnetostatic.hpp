#ifndef MACH_MAGNETOSTATIC
#define MACH_MAGNETOSTATIC

#include "mfem.hpp"
#include "adept.h"

#include "solver.hpp"
#include "coefficient.hpp"


namespace mach
{

class MagnetostaticSolver : public AbstractSolver
{
public:
	/// Class constructor.
   /// \param[in] opt_file_name - file where options are stored
   /// \param[in] smesh - if provided, defines the mesh for the problem
   /// \param[in] dim - number of dimensions
   /// \todo Can we infer dim some other way without using a template param?
   MagnetostaticSolver(const std::string &opt_file_name,
                       std::unique_ptr<mfem::Mesh> smesh = nullptr,
							  int dim = 1);

   /// Solve nonlinear magnetostatics problem using an MFEM Newton solver
   virtual void solveSteady();

private:
   // /// `bndry_marker[i]` lists the boundaries associated with a particular BC
   // std::vector<mfem::Array<int>> bndry_marker;
   // /// the mass matrix bilinear form
   // std::unique_ptr<BilinearFormType> mass;
   // /// mass matrix (move to AbstractSolver?)
   // std::unique_ptr<MatrixType> mass_matrix;

   /// Nedelec finite element collection
   std::unique_ptr<mfem::FiniteElementCollection> h_curl_coll;
   /// Raviart-Thomas finite element collection
   std::unique_ptr<mfem::FiniteElementCollection> h_div_coll;
   /// H(Curl) finite element space
   std::unique_ptr<SpaceType> h_curl_space;
   /// H(Div) finite element space
   std::unique_ptr<SpaceType> h_div_space;

   /// Magnetic vector potential A grid function
   std::unique_ptr<GridFunType> A;
   /// Magnetic flux density B = curl(A) grid function
   std::unique_ptr<GridFunType> B;

   /// the spatial residual (a semilinear form)
   std::unique_ptr<NonlinearFormType> res;

   /// current source vector
   std::unique_ptr<GridFunType> current_vec;

   /// mesh dependent reluctivity coefficient
   std::unique_ptr<MeshDependentCoefficient> nu;
   /// vector mesh dependent current density function coefficient
   std::unique_ptr<VectorMeshDependentCoefficient> current_coeff;
   /// vector mesh dependent magnetization coefficient
   std::unique_ptr<mfem::VectorCoefficient> mag_coeff;

   /// linear system solver
   std::unique_ptr<CGType> solver;
   /// linear system preconditioner
   std::unique_ptr<EMPrecType> prec;

   /// Newton solver
   mfem::NewtonSolver newton_solver;

   /// construct mesh dependent coefficient for reluctivity
   /// \param[in] alpha - used to move to lhs or rhs
   void constructReluctivity();

   /// construct mesh dependent coefficient for magnetization
   /// \param[in] alpha - used to move to lhs or rhs
   void constructMagnetization();

   /// construct vector mesh dependent coefficient for current source
   /// \param[in] alpha - used to move to lhs or rhs
   void constructCurrent();

   /// assemble vector associated with current source
   /// \note - constructCurrent must be called before calling this
   void assembleCurrentSource();

   /// TODO - implement
   /// TODO - signature will likely change because I haven't figured out how
   ///        to take derivatives yet. Probably will end up using Adept adoubles
   /// function describing nonlinear reluctivity model based on spline
   ///   interpolation of experimental B-H magnetization curve
   ///   may optionally use some temperature extrapolation
   /// \param[in] b_mag - magnitude of magnetic flux density
   static double reluctivity_model(const double b_mag);

   /// function describing current density in windings
   /// \param[in] x - position x in space of evaluation
   /// \param[out] J - current density at position x 
   static void winding_current_source(const mfem::Vector &x,
                                      mfem::Vector &J);

   /// TODO - implement function
   /// function describing permanent magnet magnetization source
   /// \param[in] x - position x in space
   /// \param[out] M - magetic flux density at position x cause by permanent
   ///                 magnets
   static void magnetization_source(const mfem::Vector &x,
                                    mfem::Vector &M);
};

} // namespace mach

#endif