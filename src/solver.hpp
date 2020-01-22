#ifndef MACH_SOLVER
#define MACH_SOLVER

#include <fstream>
#include <iostream>
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
/// dim - number of spatial dimensions (1, 2, or 3)
template <int dim>
class AbstractSolver
{
public:
   /// Class constructor.
   /// \param[in] opt_file_name - file where options are stored
   /// \param[in] smesh - if provided, defines the mesh for the problem
   AbstractSolver(const std::string &opt_file_name =
                      std::string("mach_options.json"),
                  std::unique_ptr<mfem::Mesh> smesh = nullptr);

   /// Perform set-up of derived classes using virtual functions
   /// \todo Put the constructors and this in a factory
   void initDerived();

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

   /// Initializes the state variable to a given constant
   /// \param[in] u_init - vector that defines the initial condition
   void setInitialCondition(const mfem::Vector &uic); 

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

   /// Write the mesh and residual to a vtk file
   /// \param[in] file_name - prefix file name **without** .vtk extension
   /// \param[in] refine - if >=0, indicates the number of refinements to make
   /// \todo make this work for parallel!
   /// \note the `refine` argument is useful for high-order meshes and
   /// solutions; it divides the elements up so it is possible to visualize.
   void printResidual(const std::string &file_name, int refine = -1);

   /// TODO: make this work for parallel!
   /// Write the mesh and an initializer list to a vtk file
   /// \param[in] file_name - prefix file name **without** .vtk extension
   /// \param[in] grid_funs - list of grid functions to print, passed as an
   ///                        initializer list
   /// \param[in] names - list of names to use for each grid function printed
   /// \param[in] refine - if >=0, indicates the number of refinements to make
   /// \note the `refine` argument is useful for high-order meshes and
   /// solutions; it divides the elements up so it is possible to visualize.
   void printGridFuns(const std::string &file_name,
                      std::vector<GridFunType*> grid_funs,
                      std::vector<std::string> names,
                      int refine = -1);

   /// Solve for the state variables based on current mesh, solver, etc.
   void solveForState();
   
   /// Solve for the steady state problem using newton method
   virtual void solveSteady();

   /// Solve for a transient state using a selected time-marching scheme
   virtual void solveUnsteady();
   
   /// Check the jacobian accuracy
   /// Compare the results jac_v = jac * pert_v w.r.t jac_v calculated from
   /// finite difference method 
   void jacobianCheck();

   /// set the perturbation function that used for check jacobian
   void setperturb(void (*fun)(const mfem::Vector &, mfem::Vector &))
   {  perturb_fun = fun; }
   
   /// Evaluate and return the output functional specified by `fun`
   /// \param[in] fun - specifies the desired functional
   /// \returns scalar value of estimated functional value
   double calcOutput(const std::string &fun);
   
   /// Compute the residual norm based on the current solution in `u`
   /// \returns the l2 (discrete) norm of the residual evaluated at `u`
   double calcResidualNorm();

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
   /// the spatial residual (a semilinear form)
   std::unique_ptr<NonlinearFormType> res;
   /// time-marching method (might be NULL)
   std::unique_ptr<mfem::ODESolver> ode_solver;
   /// the mass matrix bilinear form
   std::unique_ptr<BilinearFormType> mass;
   /// mass matrix
   std::unique_ptr<MatrixType> mass_matrix;
   /// TimeDependentOperator (TODO: is this the best way?)
   std::unique_ptr<mfem::TimeDependentOperator> evolver;
   /// storage for algorithmic differentiation (shared by all solvers)
   static adept::Stack diff_stack;

   /// newton solver for the steady problem
   std::unique_ptr<mfem::NewtonSolver> newton_solver;
   /// linear system solver used in newton solver
   std::unique_ptr<mfem::Solver> solver;
   /// linear system preconditioner for solver in newton solver
   std::unique_ptr<mfem::Solver> prec;
   /// `bndry_marker[i]` lists the boundaries associated with a particular BC
   std::vector<mfem::Array<int>> bndry_marker;
   /// map of output functionals
   std::map<std::string, NonlinearFormType> output;
   /// `output_bndry_marker[i]` lists the boundaries associated with output i
   std::vector<mfem::Array<int>> output_bndry_marker;
   
   /// perturbation function that used for 
   void (*perturb_fun)(const mfem::Vector &x, mfem::Vector& u);

   /// Add volume integrators to `res` based on `options`
   /// \param[in] alpha - scales the data; used to move terms to rhs or lhs
   virtual void addVolumeIntegrators(double alpha) {};

   /// Add boundary-face integrators to `res` based on `options`
   /// \param[in] alpha - scales the data; used to move terms to rhs or lhs
   virtual void addBoundaryIntegrators(double alpha) {};

   /// Add interior-face integrators to `res` based on `options`
   /// \param[in] alpha - scales the data; used to move terms to rhs or lhs
   virtual void addInterfaceIntegrators(double alpha) {};

   /// Define the number of states, the finite element space, and state u
   virtual int getNumState() {};

   /// Create `output` based on `options` and add approporiate integrators
   virtual void addOutputs() {};

};

//template <int dim>
//adept::Stack AbstractSolver<dim>::diff_stack;

} // namespace mach

#endif 
