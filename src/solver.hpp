#ifndef MACH_SOLVER
#define MACH_SOLVER

#include "mfem.hpp"
#include "utils.hpp"

using namespace std;  // TODO: needed?
using namespace mfem;

namespace mach
{

/*!
 * \class AbstractSolver
 * \brief Serves as a base class for specific PDE solvers
 */
class AbstractSolver
{
public:
   /*!
   * \brief class constructor
   * \param[in] args - contains options read in from the command line
   */
   AbstractSolver(OptionsParser &args);

   /*!
   * \brief class destructor
   */
   ~AbstractSolver();

protected:
   int num_state; ///< number of state variables at each node
   Mesh *mesh = NULL; ///< object defining the computational mesh
   ODESolver *ode_solver = NULL; ///< time-marching method (NULL if steady)
   FiniteElementCollection *fec; ///< finite element or SBP operators
   FiniteElementSpace *fes; ///< discrete function space
   BilinearForm *mass = NULL; ///< the mass matrix bilinear form
   Operator *res = NULL; ///< spatial residual form (linear in some cases)
};
    
} // namespace mach

#endif 