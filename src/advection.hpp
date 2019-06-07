#ifndef MACH_ADVECTION
#define MACH_ADVECTION

#include "mfem.hpp"
#include "solver.hpp"

namespace mach
{

/*!
* \class AdvectionIntegrator
* \brief Linear advection integrator specialized to SBP operators
*/
class AdvectionIntegrator : public mfem::BilinearFormIntegrator
{
public:

   /*!
   * \brief Constructs a linear advection integrator
   * \param[in] velc - represents the (possibly) spatially varying velocity
   * \param[in] alpha - scales the terms; can be used to move from lhs to rhs
   */
   AdvectionIntegrator(mfem::VectorCoefficient &velc, double a = 1.0)
      : vel_coeff(velc) { alpha = a; }

   /*!
   * \brief Create the element stiffness matrix for linear advection
   * \param[in] el - the finite element whose stiffness matrix we want
   * \param[in] Trans - defines the reference to physical element mapping
   * \param[out] elmat - the desired element stiffness matrix
   */
   virtual void AssembleElementMatrix(const mfem::FiniteElement &el,
                                      mfem::ElementTransformation &Trans,
                                      mfem::DenseMatrix &elmat);

private:
#ifndef MFEM_THREAD_SAFE
   /// velocity in physical space
   mfem::DenseMatrix vel;
   /// scaled velocity in reference space
   mfem::DenseMatrix velhat;
   /// adjJ = |J|*dxi/dx = adj(dx/dxi)
   mfem::DenseMatrix adjJ;
   /// Storage for derivative operators
   mfem::DenseMatrix D;
   /// Storage for the diagonal norm matrix
   mfem::Vector H;
   /// reference to one component of velhat at all nodes
   mfem::Vector Udi;
#endif
   /// represents the (possibly) spatially varying velocity field
   mfem::VectorCoefficient &vel_coeff;
   /// scales the terms; can be used to move to rhs/lhs
   double alpha;
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