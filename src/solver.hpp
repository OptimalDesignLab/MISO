#ifndef MACH_SOLVER
#define MACH_SOLVER

#include "mfem.hpp"
#include "adept.h"
#include "mach_types.hpp"
#include "utils.hpp"
#include "json.hpp"
#ifdef MFEM_USE_SIMMETRIX
#include <SimUtil.h>
#include <gmi_sim.h>
#endif
#ifdef MFEM_USE_PUMI
#include <apfMDS.h>
#include <gmi_null.h>
#include <PCU.h>
#include <apfConvert.h>
#include <gmi_mesh.h>
#include <crv.h>
#endif
namespace mach
{

/// Serves as a base class for specific PDE solvers
class AbstractSolver
{
   
public:
   /// Class constructor.
   /// \param[in] opt_file_name - file where options are stored
   /// \param[in] smesh - if provided, defines the mesh for the problem
   AbstractSolver(const std::string &opt_file_name =
                      std::string("mach_options.json"),
                  std::unique_ptr<mfem::Mesh> smesh = nullptr);

   /// class destructor
   ~AbstractSolver();

   /// Constructs the mesh member based on c preprocesor defs
   /// \param[in] smesh - if provided, defines the mesh for the problem
   void constructMesh(std::unique_ptr<mfem::Mesh> smesh = nullptr);

   /// Initializes the state variable to a given function.
   /// \param[in] u_init - function that defines the initial condition
   /// \note The second argument in the function `u_init` is the initial condition
   /// value.  This may be a vector of length 1 for scalar.
   void setInitialCondition(void (*u_init)(const mfem::Vector &,
                                           mfem::Vector &));

   /// Returns the L2 error between the state `u` and given exact solution.
   /// \param[in] u_exact - function that defines the exact solution
   /// \param[in] entry - if >= 0, the L2 error of state `entry` is returned
   /// \returns L2 error
   double calcL2Error(void (*u_exact)(const mfem::Vector &, mfem::Vector &),
                      int entry = -1);

   /// Find the gobal step size for the given CFL number
   /// \param[in] cfl - target CFL number for the domain
   /// \returns dt_min - the largest step size for the given CFL
   virtual double calcStepSize(double cfl) const;

   /// Write the mesh and solution to a vtk file
   /// \param[in] file_name - prefix file name **without** .vtk extension
   /// \param[in] refine - if >=0, indicates the number of refinements to make
   /// \todo make this work for parallel!
   /// \note the `refine` argument is useful for high-order meshes and
   /// solutions; it divides the elements up so it is possible to visualize.
   void printSolution(const std::string &file_name, int refine = -1);

   /// Solve for the state variables based on current mesh, solver, etc.
   void solveForState();

protected:
#ifdef MFEM_USE_MPI
   /// communicator used by MPI group for communication
   MPI_Comm comm;
#endif
   /// process rank
   int rank;
   /// print object
   std::ostream *out;
   /// solver options
   nlohmann::json options;
   /// number of space dimensions
   int num_dim;
   /// number of state variables at each node
   int num_state = 0;
   /// time step size
   double dt;
   /// final time
   double t_final;
   /// pumi mesh object
#ifdef MFEM_USE_PUMI
   apf::Mesh2* pumi_mesh;
#endif
   /// finite element or SBP operators
   std::unique_ptr<mfem::FiniteElementCollection> fec;
   /// object defining the computational mesh
   std::unique_ptr<MeshType> mesh;
   /// discrete function space
   std::unique_ptr<SpaceType> fes;
   /// state variable
   std::unique_ptr<GridFunType> u;
   /// time-marching method (might be NULL)
   std::unique_ptr<mfem::ODESolver> ode_solver;
   /// the mass matrix bilinear form
   //std::unique_ptr<MassFormType> mass;
   /// operator for spatial residual (linear in some cases)
   //std::unique_ptr<ResFormType> res;
   /// TimeDependentOperator (TODO: is this the best way?)
   std::unique_ptr<mfem::TimeDependentOperator> evolver;
   /// storage for algorithmic differentiation (shared by all solvers)
   static adept::Stack diff_stack;
};

} // namespace mach

#endif 
