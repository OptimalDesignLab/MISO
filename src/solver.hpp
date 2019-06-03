#ifndef MACH_SOLVER
#define MACH_SOLVER

#include "mfem.hpp"
#include "utils.hpp"

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
   AbstractSolver(mfem::OptionsParser &args);

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
   void set_initial_condition(
      void (*u_init)(const mfem::Vector &, mfem::Vector &));

protected:

   /// number of state variables at each node
   int num_state = 0;

   /// state variable
   std::unique_ptr<mfem::GridFunction> u; 

   /// object defining the computational mesh
   std::unique_ptr<mfem::Mesh> mesh;

   /// time-marching method (might be NULL)
   std::unique_ptr<mfem::ODESolver> ode_solver;

   /// finite element or SBP operators
   std::unique_ptr<mfem::FiniteElementCollection> fec;

   /// discrete function space
   std::unique_ptr<mfem::FiniteElementSpace> fes;

   /// the mass matrix bilinear form
   std::unique_ptr<mfem::BilinearForm> mass;

   /// operator for spatial residual (linear in some cases)
   std::unique_ptr<mfem::Operator> res;
};
    
} // namespace mach

#endif 