#ifndef MACH_ADVECTION
#define MACH_ADVECTION

#include "mfem.hpp"
#include "solver.hpp"

namespace mach
{

/*!
 * \class AdvectionIntegrator
 * \brief linear advection integrator specialized to SBP operators
 */
class AdvectionIntegrator : public mfem::BilinearFormIntegrator
{
private:
#ifndef MFEM_THREAD_SAFE
   mfem::DenseMatrix dshape, adjJ, Q_ir;
   mfem::Vector shape, vec2, BdFidxT;
#endif
   mfem::VectorCoefficient &Q;
   double alpha;

public:
   AdvectionIntegrator(mfem::VectorCoefficient &q, double a = 1.0)
      : Q(q) { alpha = a; }
   virtual void AssembleElementMatrix(const mfem::FiniteElement &,
                                      mfem::ElementTransformation &,
                                      mfem::DenseMatrix &);
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
   AdvectionSolver(mfem::OptionsParser &args,
                   void (*vel_field)(const mfem::Vector &, mfem::Vector &));

protected:

   /// the velocity field
   std::unique_ptr<mfem::VectorFunctionCoefficient> velocity;

};
    
} // namespace mach

#endif 