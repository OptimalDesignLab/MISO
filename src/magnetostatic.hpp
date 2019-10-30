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
   std::unique_ptr<mfem::VectorCoefficient> current_coeff;

   /// construct mesh dependent coefficient for reluctivity
   /// \param[in] alpha - used to move to lhs or rhs
   void constructReluctivity(double alpha);

   /// construct vector mesh dependent coefficient for current source
   /// \param[in] alpha - used to move to lhs or rhs
   void constructCurrent(double alpha);

   /// assemble vector associated with current source
   /// \note - constructCurrent must be called before calling this
   void assembleCurrentSource();
};

} // namespace mach

#endif