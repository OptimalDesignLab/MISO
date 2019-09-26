#ifndef MACH_MAGNETOSTATIC
#define MACH_MAGNETOSTATIC

#include "mfem.hpp"
#include "adept.h"

#include "solver.hpp"


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

protected:
   /// `bndry_marker[i]` lists the boundaries associated with a particular BC
   std::vector<mfem::Array<int>> bndry_marker;
   /// the mass matrix bilinear form
   std::unique_ptr<BilinearFormType> mass;
   /// the spatial residual (a semilinear form)
   std::unique_ptr<NonlinearFormType> res;
   /// mass matrix (move to AbstractSolver?)
   std::unique_ptr<MatrixType> mass_matrix;
};

} // namespace mach

#endif