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

   /*!
   * \brief initialize the state variable to given function
   * \param[in] u_init - function that defines the velocity field
   * 
   * \note The second argument in the function u0 is the initial condition
   * value.  This may be a vector of length 1 for scalar.
   */
   void set_initial_condition(void (*u_init)(const Vector &, Vector &));

protected:
   int num_state = 0; ///< number of state variables at each node
   GridFunction *u; ///< state variable 
   Mesh *mesh = NULL; ///< object defining the computational mesh
   ODESolver *ode_solver = NULL; ///< time-marching method (NULL if steady)
   FiniteElementCollection *fec; ///< finite element or SBP operators
   FiniteElementSpace *fes; ///< discrete function space
   BilinearForm *mass = NULL; ///< the mass matrix bilinear form
   Operator *res = NULL; ///< spatial residual form (linear in some cases)
};
    
} // namespace mach

#endif 