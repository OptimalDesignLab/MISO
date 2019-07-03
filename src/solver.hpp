#ifndef MACH_SOLVER
#define MACH_SOLVER

#include "mfem.hpp"
#include "mach_types.hpp"
#include "utils.hpp"
#include "json.hpp"

namespace mach
{

/// Serves as a base class for specific PDE solvers
class AbstractSolver
{
   
public:
   /// Class constructor.
   /// \param[in] opt_file_name - file where options are stored
  
   AbstractSolver(const std::string &opt_file_name =
                      std::string("mach_options.json"));

   /// class destructor
   ~AbstractSolver();

   /// Initializes the state variable to a given function.
   /// \param[in] u_init - function that defines the initial condition
   /// \note The second argument in the function `u_init` is the initial condition
   /// value.  This may be a vector of length 1 for scalar.
   void setInitialCondition(void (*u_init)(const mfem::Vector &,
                                           mfem::Vector &));

   /// Returns the L2 error between the state `u` and given exact solution.
   /// \param[in] u_exact - function that defines the exact solution
   /// \returns L2 error
   double calcL2Error(void (*u_exact)(const mfem::Vector &, mfem::Vector &));

   /// Solve for the state variables based on current mesh, solver, etc.
   void solveForState();

protected:
#ifdef MFEM_USE_MPI
   /// communicator used by MPI group for communication
   MPI_Comm comm;
   /// process rank
   int rank;
#endif
   /// solver options
   nlohmann::json options;
   /// number of state variables at each node
   int num_state = 0;
   /// time step size
   double dt;
   /// final time
   double t_final;
   /// finite element or SBP operators
   std::unique_ptr<mfem::FiniteElementCollection> fec;
   /// object defining the computational mesh
   std::unique_ptr<mfem::Mesh> mesh;
   /// discrete function space
   std::unique_ptr<mfem::FiniteElementSpace> fes;
   /// state variable
   std::unique_ptr<mfem::GridFunction> u;
   /// time-marching method (might be NULL)
   std::unique_ptr<mfem::ODESolver> ode_solver;
   /// the mass matrix bilinear form
   std::unique_ptr<mfem::BilinearForm> mass;
   /// operator for spatial residual (linear in some cases)
   std::unique_ptr<mfem::Operator> res;
   /// TimeDependentOperator (TODO: is this the best way?)
   std::unique_ptr<mfem::TimeDependentOperator> evolver;
};
    
} // namespace mach

#endif 
