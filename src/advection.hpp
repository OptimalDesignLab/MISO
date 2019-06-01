#ifndef MACH_ADVECTION
#define MACH_ADVECTION

#include "mfem.hpp"
#include "solver.hpp"

using namespace std;  // TODO: needed?
using namespace mfem;

namespace mach
{

/*!
 * \class AdvectionIntegrator
 * \brief linear advection integrator specialized to SBP operators
 */
class AdvectionIntegrator : public BilinearFormIntegrator
{
private:
#ifndef MFEM_THREAD_SAFE
   DenseMatrix dshape, adjJ, Q_ir;
   Vector shape, vec2, BdFidxT;
#endif
   VectorCoefficient &Q;
   double alpha;

public:
   AdvectionIntegrator(VectorCoefficient &q, double a = 1.0)
      : Q(q) { alpha = a; }
   virtual void AssembleElementMatrix(const FiniteElement &,
                                      ElementTransformation &,
                                      DenseMatrix &);
};

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