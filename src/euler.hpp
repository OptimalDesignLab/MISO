#ifndef MACH_EULER
#define MACH_EULER

#include "mfem.hpp"
#include "solver.hpp"

namespace mach
{

/// Solver for linear advection problems
class EulerSolver : public AbstractSolver
{
public:
   /// Class constructor.
   /// \param[in] opt_file_name - file where options are stored
   /// \param[in] dim - number of dimensions
   /// \todo Can we infer dim some other way?
   EulerSolver(const std::string &opt_file_name, int dim = 1);

protected:
   /// the mass matrix bilinear form
   std::unique_ptr<BilinearFormType> mass;
   /// the spatial residual (a semilinear form)
   std::unique_ptr<NonlinearFormType> res;
   /// mass matrix (move to AbstractSolver?)
   std::unique_ptr<MatrixType> mass_matrix;
};

} // namespace mach

#endif