#ifndef MACH_ADVECTION
#define MACH_ADVECTION

#include "mfem.hpp"
#include "solver.hpp"

using namespace std;  // TODO: needed?
using namespace mfem;

namespace mach
{

/*!
 * \class AdvectionSolver
 * \brief Solver for linear advection problems
 */
class AdvectionSolver : public AbstractSolver
{
public:
   /*!
   * \brief class constructor
   * \param[in] args - contains options read in from the command line
   * \param[in] vel_field - function that defines the velocity field
   */
   AdvectionSolver(OptionsParser &args,
                   void (*vel_field)(const Vector &, Vector &));

protected:
   VectorFunctionCoefficient *velocity = NULL; ///< the velocity field

};
    
} // namespace mach

#endif 