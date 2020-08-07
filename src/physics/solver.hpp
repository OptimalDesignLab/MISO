#ifndef MACH_SOLVER
#define MACH_SOLVER

#include <fstream>
#include <iostream>
#include <functional>

#include "adept.h"
#include "json.hpp"
#include "mfem.hpp"

#include "mach_types.hpp"
#include "utils.hpp"

#ifdef MFEM_USE_PUMI
namespace apf
{
class Mesh2;
} // namespace apf
#include "PCU.h"
#ifdef MFEM_USE_SIMMETRIX
#include "SimUtil.h"
#include "gmi_sim.h"
#endif // MFEM_USE_SIMMETRIX

#ifdef MFEM_USE_EGADS
#include "gmi_egads.h"
#endif // MFEM_USE_EGADS
namespace mach
{
struct pumiDeleter
{
   void operator()(apf::Mesh2* mesh) const
   {
      mesh->destroyNative();
      apf::destroyMesh(mesh);
      PCU_Comm_Free();
#ifdef MFEM_USE_SIMMETRIX
      gmi_sim_stop();
      Sim_unregisterAllKeys();
#endif // MFEM_USE_SIMMETRIX

#ifdef MFEM_USE_EGADS
      gmi_egads_stop();
#endif // MFEM_USE_EGADS
   }
};

} // namespace mach} // namespace mach
#endif

namespace mach
{

class MachEvolver;

/// Serves as a base class for specific PDE solvers
class AbstractSolver
{
public:
   /// Class constructor.
   /// \param[in] opt_file_name - file where options are stored
   /// \param[in] smesh - if provided, defines the mesh for the problem
   // AbstractSolver(const std::string &opt_file_name,
   //                std::unique_ptr<mfem::Mesh> smesh);

   /// Class constructor.
   /// \param[in] options - pre-loaded JSON options object
   /// \param[in] smesh - if provided, defines the mesh for the problem
   /// \param[in] comm - MPI communicator for parallel operations
   AbstractSolver(const nlohmann::json &options,
                  std::unique_ptr<mfem::Mesh> smesh,
                  MPI_Comm comm);

   /// Perform set-up of derived classes using virtual functions
   /// \todo Put the constructors and this in a factory
   virtual void initDerived();

   /// class destructor
   virtual ~AbstractSolver();

   /// TODO: should this be pretected/private?
   /// Constructs the mesh member based on c preprocesor defs
   /// \param[in] smesh - if provided, defines the mesh for the problem
   void constructMesh(std::unique_ptr<mfem::Mesh> smesh);

   /// Initializes the state vector to a given scalar function.
   /// \param[in] u_init - function that defines the initial condition
   inline virtual void setInitialCondition(
      const std::function<double(const mfem::Vector &)> &u_init)
   { setInitialCondition(*u, u_init); };

   /// Initializes the state vector to a given function.
   /// \param[in] u_init - function that defines the initial condition
   /// \note The second argument in the function `u_init` is the initial condition
   /// value.  This may be a vector of length 1 for scalar.
   inline virtual void setInitialCondition(
      const std::function<void(const mfem::Vector &, mfem::Vector &)> &u_init)
   { setInitialCondition(*u, u_init); };

   /// Initializes the state variable to a given constant
   /// \param[in] u_init - value that defines the initial condition
   inline virtual void setInitialCondition(const double u_init)
   { setInitialCondition(*u, u_init); };

   /// Initializes the state variable to a given constant vector
   /// \param[in] u_init - vector that defines the initial condition
   inline virtual void setInitialCondition(const mfem::Vector &u_init)
   { setInitialCondition(*u, u_init); };

   /// Initializes the state vector to a given scalar function.
   /// \param[in] state - the state vector to initialize
   /// \param[in] u_init - function that defines the initial condition
   virtual void setInitialCondition(
      mfem::ParGridFunction &state,
      const std::function<double(const mfem::Vector &)> &u_init);

   /// Initializes the state vector to a given function.
   /// \param[in] state - the state vector to initialize
   /// \param[in] u_init - function that defines the initial condition
   /// \note The second argument in the function `u_init` is the initial condition
   /// value.  This may be a vector of length 1 for scalar.
   virtual void setInitialCondition(
      mfem::ParGridFunction &state,
      const std::function<void(const mfem::Vector &, mfem::Vector &)> &u_init);

   /// Initializes the state variable to a given constant
   /// \param[in] state - the state vector to initialize
   /// \param[in] u_init - value that defines the initial condition
   virtual void setInitialCondition(
      mfem::ParGridFunction &state,
      const double u_init);

   /// Initializes the state variable to a given constant vector
   /// \param[in] state - the state vector to initialize
   /// \param[in] u_init - vector that defines the initial condition
   virtual void setInitialCondition(
      mfem::ParGridFunction &state,
      const mfem::Vector &u_init);

   /// TODO move to protected?
   /// Returns the integral inner product between two grid functions
   /// \param[in] x - grid function 
   /// \param[in] y - grid function 
   /// \return integral inner product between `x` and `y`
   double calcInnerProduct(const GridFunType &x, const GridFunType &y);

   /// Returns the L2 error between the state `u` and given exact solution.
   /// \param[in] u_exact - function that defines the exact solution
   /// \return L2 error
   double calcL2Error(const std::function<double(const mfem::Vector &)> &u_exact);

   /// Returns the L2 error between the state `u` and given exact solution.
   /// \param[in] u_exact - function that defines the exact solution
   /// \param[in] entry - if >= 0, the L2 error of state `entry` is returned
   /// \return L2 error
   double calcL2Error(const std::function<void(const mfem::Vector &,
                                               mfem::Vector &)> &u_exact,
                      int entry = -1);

   /// Returns the L2 error of a field and given exact solution.
   /// \param[in] field - grid function to compute L2 error for
   /// \param[in] u_exact - function that defines the exact solution
   /// \param[in] entry - if >= 0, the L2 error of state `entry` is returned
   /// \return L2 error
   double calcL2Error(GridFunType *field,
                      const std::function<double(const mfem::Vector &)> &u_exact);
   
   /// Returns the L2 error of a field and given exact solution.
   /// \param[in] field - grid function to compute L2 error for
   /// \param[in] u_exact - function that defines the exact solution
   /// \param[in] entry - if >= 0, the L2 error of state `entry` is returned
   /// \return L2 error
   double calcL2Error(GridFunType *field,
                      const std::function<void(const mfem::Vector &,
                                               mfem::Vector &)> &u_exact,
                      int entry = -1);

   /// Find the step size based on the options
   /// \param[in] iter - the current iteration
   /// \param[in] t - the current time (before the step)
   /// \param[in] t_final - the final time
   /// \param[in] dt_old - the step size that was just taken
   /// \param[in] state - the current state
   /// \returns dt - the step size appropriate to the problem
   /// This base method simply returns the option in ["time-dis"]["dt"],
   /// truncated as necessary such that `t + dt = t_final`.
   virtual double calcStepSize(int iter, double t, double t_final,
                               double dt_old,
                               const mfem::ParGridFunction &state) const;

   /// Write the mesh and solution to a vtk file
   /// \param[in] file_name - prefix file name **without** .vtk extension
   /// \param[in] refine - if >=0, indicates the number of refinements to make
   /// \todo make this work for parallel!
   /// \note the `refine` argument is useful for high-order meshes and
   /// solutions; it divides the elements up so it is possible to visualize.
   virtual void printSolution(const std::string &file_name, int refine = -1);

   /// Write the mesh and adjoint to a vtk file
   /// \param[in] file_name - prefix file name **without** .vtk extension
   /// \param[in] refine - if >=0, indicates the number of refinements to make
   /// \todo make this work for parallel!
   /// \note the `refine` argument is useful for high-order meshes and
   /// solutions; it divides the elements up so it is possible to visualize.
   void printAdjoint(const std::string &file_name, int refine = -1);

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
   /// \param[in] field - grid function to print
   /// \param[in] name - name to use for the printed grid function
   /// \param[in] refine - if >=0, indicates the number of refinements to make
   /// \note the `refine` argument is useful for high-order meshes and
   /// solutions; it divides the elements up so it is possible to visualize.
   void printField(const std::string &file_name,
                   mfem::ParGridFunction &field,
                   const std::string &name,
                   int refine = -1)
   { printFields(file_name, {&field}, {name}, refine); };

   /// TODO: make this work for parallel!
   /// Write the mesh and an initializer list to a vtk file
   /// \param[in] file_name - prefix file name **without** .vtk extension
   /// \param[in] fields - list of grid functions to print, passed as an
   ///                     initializer list
   /// \param[in] names - list of names to use for each grid function printed
   /// \param[in] refine - if >=0, indicates the number of refinements to make
   /// \note the `refine` argument is useful for high-order meshes and
   /// solutions; it divides the elements up so it is possible to visualize.
   void printFields(const std::string &file_name,
                      std::vector<mfem::ParGridFunction *> fields,
                      std::vector<std::string> names,
                      int refine = -1);

   /// \brief Returns a vector of pointers to grid functions that define fields
   /// Default behavior is to return just the state `u`
   virtual std::vector<GridFunType*> getFields();

   /// Solve for the state variables based on current mesh, solver, etc.
   virtual void solveForState() { solveForState(*u); };

   virtual void solveForState(mfem::ParGridFunction &state);
   
   /// Solve for the adjoint based on current mesh, solver, etc.
   /// \param[in] fun - specifies the functional corresponding to the adjoint
   void solveForAdjoint(const std::string &fun);

   /// Check the Jacobian using a finite-difference directional derivative
   /// \param[in] pert - function that defines the perturbation direction
   /// \note Compare the results of the project Jac*pert using the Jacobian
   /// directly versus a finite-difference based product.  
   void checkJacobian(void (*pert_fun)(const mfem::Vector &, mfem::Vector &));
   
   /// Evaluate and return the output functional specified by `fun`
   /// \param[in] fun - specifies the desired functional
   /// \returns scalar value of estimated functional value
   double calcOutput(const std::string &fun)
   { return calcOutput(*u, fun); };

   /// Evaluate and return the output functional specified by `fun`
   /// \param[in] state - the state vector to evaluate the functional at
   /// \param[in] fun - specifies the desired functional
   /// \returns scalar value of estimated functional value
   double calcOutput(const mfem::ParGridFunction &state,
                     const std::string &fun);
   
   /// Compute the residual norm based on the current solution in `u`
   /// \returns the l2 (discrete) norm of the residual evaluated at `u`
   double calcResidualNorm() const { return calcResidualNorm(*u); };

   /// Compute the residual norm based on the input `state`
   /// \returns the l2 (discrete) norm of the residual evaluated at `u`
   double calcResidualNorm(const mfem::ParGridFunction &state) const;

   /// Return a state sized vector constructed from an externally allocated array
   /// \param[in] data - external data array
   /// \note If `data` is nullptr a new array will be allocated. If `data` is 
   /// not `nullptr` it is assumed to be of size of at least `fes->GetVSize()`
   std::unique_ptr<mfem::ParGridFunction> getNewField(double *data = nullptr);

   /// Compute the residual based on the current solution in `u`
   /// \param[out] residual - the residual
   void calcResidual(mfem::ParGridFunction &residual)
   { calcResidual(*u, residual); };

   /// Compute the residual based on `state` and store the it in `residual`
   /// \param[in] state - the current state to evaluate the residual at
   /// \param[out] residual - the residual
   void calcResidual(const mfem::ParGridFunction &state, 
                     mfem::ParGridFunction &residual);
   
   /// TODO: Who added this?  Do we need it still?  What is it for?  Document!
   void feedpert(void (*p)(const mfem::Vector &, mfem::Vector &)) { pert = p; }

   /// Return the output map
   std::map<std::string, NonlinearFormType> GetOutput() const { return output; }

   /// convert conservative variables to entropy variables
   /// \param[in/out] state - the conservative/entropy variables
   //virtual void convertToEntvar(mfem::Vector &state) { };

   /// Return a pointer to the solver's mesh
   MeshType* getMesh() {return mesh.get();}

   void printMesh(const std::string &filename)
   { mesh->PrintVTU(filename, mfem::VTKFormat::BINARY, true, 0); }
   /// return a reference to the mesh's coordinate field
   mfem::Vector& getMeshCoordinates() { return *mesh->GetNodes(); }

   /// \brief function to update the mesh's nodal coordinate field
   /// \param[in] coords - Vector containing mesh's nodal coordinate field
   /// \note the size of `coords` is assumed to be the size returned from the
   /// mesh finite element space's vdof size
   /// \note After calling this method the mesh will own the GridFunction
   /// defining the coordinate field, but the GridFunction will not own the 
   /// underlying data (TODO? Look at cost of copying?)
   void setMeshCoordinates(mfem::Vector &coords);

   inline int getMeshSize() { return mesh->GetNodes()->FESpace()->GetVSize(); }
   inline int getStateSize() { return fes->GetVSize(); }

#ifdef MFEM_USE_PUMI
   /// Return a pointer to the underlying PUMI mesh
   apf::Mesh2* getPumiMesh() {return pumi_mesh.get();};
#endif

protected:
   /// communicator used by MPI group for communication
   MPI_Comm comm;
   /// process rank
   int rank;
   /// print object
   std::ostream *out;
   /// solver options
   nlohmann::json options;
   /// material Library
   nlohmann::json materials;
   /// number of state variables at each node
   int num_state = 0;
   /// time step size
   double dt;
   /// final time
   double t_final;
   
   //--------------------------------------------------------------------------
   // Members associated with the mesh
   /// object defining the mfem computational mesh
   std::unique_ptr<MeshType> mesh;
#ifdef MFEM_USE_PUMI
   /// pumi mesh object
   // apf::Mesh2* pumi_mesh;
   std::unique_ptr<apf::Mesh2, pumiDeleter> pumi_mesh;
#endif

   //--------------------------------------------------------------------------
   // Members associated with fields
   /// finite element or SBP operators
   std::unique_ptr<mfem::FiniteElementCollection> fec;
   /// discrete finite element space
   std::unique_ptr<SpaceType> fes;
   /// state variable
   std::unique_ptr<GridFunType> u;
   /// adjoint variable 
   std::unique_ptr<GridFunType> adj;
   /// derivative of L = J + psi^T res, with respect to mesh nodes
   std::unique_ptr<GridFunType> dLdX;

   //--------------------------------------------------------------------------
   // Members associated with forms
   /// the nonlinear form evaluate the mass matrix
   std::unique_ptr<NonlinearFormType> nonlinear_mass;
   /// the mass matrix bilinear form
   std::unique_ptr<BilinearFormType> mass;
   /// the spatial residual (a semilinear form)
   std::unique_ptr<NonlinearFormType> res;
   /// the stiffness matrix bilinear form
   std::unique_ptr<BilinearFormType> stiff;
   /// the load vector linear form
   std::unique_ptr<LinearFormType> load;
   /// entropy/energy that is needed for RRK methods
   std::unique_ptr<NonlinearFormType> ent;

   /// derivative of psi^T res w.r.t the mesh nodes
   std::unique_ptr<NonlinearFormType> res_mesh_sens;

   /// storage for algorithmic differentiation (shared by all solvers)
   static adept::Stack diff_stack;

   //--------------------------------------------------------------------------
   // Members associated with time marching (and Newton's method)
   /// time-marching method (might be NULL)
   std::unique_ptr<mfem::ODESolver> ode_solver;
   /// the operator used for time-marching ODEs 
   std::unique_ptr<MachEvolver> evolver;

   /// newton solver for the steady problem
   std::unique_ptr<mfem::NewtonSolver> newton_solver;
   /// linear system solver used in newton solver
   std::unique_ptr<mfem::Solver> solver;
   /// linear system preconditioner for solver in newton solver and adjoint
   std::unique_ptr<mfem::Solver> prec;

   //--------------------------------------------------------------------------
   // Members associated with boundary conditions and outputs
   /// Array that marks boundaries as essential
   mfem::Array<int> ess_bdr;
   /// `bndry_marker[i]` lists the boundaries associated with a particular BC
   std::vector<mfem::Array<int>> bndry_marker;
   /// map of output functionals
   std::map<std::string, NonlinearFormType> output;
   /// `output_bndry_marker[i]` lists the boundaries associated with output i
   std::vector<mfem::Array<int>> output_bndry_marker;

   //--------------------------------------------------------------------------

   /// Construct PUMI Mesh
   void constructPumiMesh();

   /// Construct various coefficients
   virtual void constructCoefficients() {};

   /// Initialize all forms needed by the derived class
   /// \note Derived classes must allocate the forms they need to use.  Only
   /// allocated forms will have integrators added to them.
   virtual void constructForms()
   {
      throw MachException("constructForms() not defined by derived class!");
   };

   /// Add domain integrators to `mass`
   /// \param[in] alpha - scales the data; used to move terms to rhs or lhs
   virtual void addMassIntegrators(double alpha);

   /// Add domain integrators to `nonlinear_mass`
   /// \param[in] alpha - scales the data; used to move terms to rhs or lhs
   virtual void addNonlinearMassIntegrators(double alpha) {};

   /// Add volume integrators to `res` based on `options`
   /// \param[in] alpha - scales the data; used to move terms to rhs or lhs
   virtual void addResVolumeIntegrators(double alpha) {};

   /// Add boundary-face integrators to `res` based on `options`
   /// \param[in] alpha - scales the data; used to move terms to rhs or lhs
   virtual void addResBoundaryIntegrators(double alpha) {};

   /// Add interior-face integrators to `res` based on `options`
   /// \param[in] alpha - scales the data; used to move terms to rhs or lhs
   virtual void addResInterfaceIntegrators(double alpha) {};

   /// Add volume integrators to `stiff`
   /// \param[in] alpha - scales the data; used to move terms to rhs or lhs
   virtual void addStiffVolumeIntegrators(double alpha) {};

   /// Add boundary-face integrators to `stiff`
   /// \param[in] alpha - scales the data; used to move terms to rhs or lhs
   virtual void addStiffBoundaryIntegrators(double alpha) {};

   /// Add interior-face integrators to `stiff`
   /// \param[in] alpha - scales the data; used to move terms to rhs or lhs
   virtual void addStiffInterfaceIntegrators(double alpha) {};

   /// Add volume integrators to 'load'
   /// \param[in] alpha - scales the data; used to move terms to rhs or lhs
   virtual void addLoadVolumeIntegrators(double alpha) {};

   /// Add boundary-face integrators to `load'
   /// \param[in] alpha - scales the data; used to move terms to rhs or lhs
   virtual void addLoadBoundaryIntegrators(double alpha) {};

   /// Add interior-face integrators to `load'
   /// \param[in] alpha - scales the data; used to move terms to rhs or lhs
   virtual void addLoadInterfaceIntegrators(double alpha) {};

   /// Add volume integrators for `ent`
   virtual void addEntVolumeIntegrators() {};

   /// mark which boundaries are essential
   virtual void setEssentialBoundaries();

   /// Define the number of states, the finite element space, and state u
   virtual int getNumState() = 0; 

   /// Create `output` based on `options` and add approporiate integrators
   virtual void addOutputs() {};

   /// Solve for the steady state problem using newton method
   virtual void solveSteady(mfem::ParGridFunction &state);

   /// Solve for a transient state using a selected time-marching scheme
   virtual void solveUnsteady(mfem::ParGridFunction &state);
   
   /// For code that should be executed before the time stepping begins
   /// \param[in] state - the current state
   virtual void initialHook(const mfem::ParGridFunction &state) {};

   /// For code that should be executed before `ode_solver->Step`
   /// \param[in] iter - the current iteration
   /// \param[in] t - the current time (before the step)
   /// \param[in] dt - the step size that will be taken
   /// \param[in] state - the current state
   virtual void iterationHook(int iter, double t, double dt,
                              const mfem::ParGridFunction &state) {};

   /// Determines when to exit the time stepping loop
   /// \param[in] iter - the current iteration
   /// \param[in] t - the current time (after the step)
   /// \param[in] t_final - the final time
   /// \param[in] dt - the step size that was just taken
   /// \param[in] state - the current state
   virtual bool iterationExit(int iter, double t, double t_final, double dt,
                              const mfem::ParGridFunction &state );

   /// For code that should be executed after the time stepping ends
   /// \param[in] iter - the terminal iteration
   /// \param[in] t_final - the final time
   /// \param[in] state - the current state
   virtual void terminalHook(int iter, double t_final,
                             const mfem::ParGridFunction &state) {};
      
   /// Solve for a steady adjoint
   /// \param[in] fun - specifies the functional corresponding to the adjoint
   virtual void solveSteadyAdjoint(const std::string &fun);

   /// Solve for an unsteady adjoint
   /// \param[in] fun - specifies the functional corresponding to the adjoint
   virtual void solveUnsteadyAdjoint(const std::string &fun);

   /// TODO: What is this doing here?
   void (*pert)(const mfem::Vector &, mfem::Vector &);

   /// Constuct the linear system solver
   /// \note solver and preconditioner chosen based on options
   virtual void constructLinearSolver(nlohmann::json &options);

   /// Constructs the newton solver object
   virtual void constructNewtonSolver();

   /// Sets convergence options for solver
   /// \param[in] options - options structure for particular solver to set
   virtual void setIterSolverOptions(nlohmann::json &options);

   /// Constructs the operator that defines ODE evolution 
   virtual void constructEvolver();

   /// Used by derived classes that themselves construct solver objects that
   /// don't need all the memory for a fully featured solver, that just need to
   /// support the AbstractSolver interface (JouleSolver)
   AbstractSolver(const std::string &opt_file_name,
                  MPI_Comm comm = MPI_COMM_WORLD);

private:
   /// explicitly prohibit copy construction
   AbstractSolver(const AbstractSolver&) = delete;
   AbstractSolver& operator=(const AbstractSolver&) = delete;

   /// Used to do the bulk of the initialization shared between constructors
   /// \param[in] options - pre-loaded JSON options object
   /// \param[in] smesh - if provided, defines the mesh for the problem
   /// \param[in] comm - MPI communicator to use for parallel operations
   void initBase(const nlohmann::json &file_options,
                 std::unique_ptr<mfem::Mesh> smesh,
                 MPI_Comm comm);

};

using SolverPtr = std::unique_ptr<AbstractSolver>;

/// Creates a new `DerivedSolver` and initializes it
/// \param[in] json_options - json object that stores options
/// \param[in] smesh - if provided, defines the mesh for the problem
/// \param[in] comm - MPI communicator for parallel operations
/// \tparam DerivedSolver - a derived class of `AbstractSolver`
template <class DerivedSolver>
SolverPtr createSolver(const nlohmann::json &json_options,
                       std::unique_ptr<mfem::Mesh> smesh = nullptr,
                       MPI_Comm comm = MPI_COMM_WORLD)
{
   //auto solver = std::make_unique<DerivedSolver>(opt_file_name, move(smesh));
   SolverPtr solver(new DerivedSolver(json_options, move(smesh), comm));
   solver->initDerived();
   return solver;
}

/// Creates a new `DerivedSolver` and initializes it
/// \param[in] opt_file_name - file where options are stored
/// \param[in] smesh - if provided, defines the mesh for the problem
/// \param[in] comm - MPI communicator for parallel operations
/// \tparam DerivedSolver - a derived class of `AbstractSolver`
template <class DerivedSolver>
SolverPtr createSolver(const std::string &opt_file_name,
                       std::unique_ptr<mfem::Mesh> smesh = nullptr,
                       MPI_Comm comm = MPI_COMM_WORLD)
{
   nlohmann::json json_options;
   std::ifstream options_file(opt_file_name);
   options_file >> json_options;
   return createSolver<DerivedSolver>(json_options, move(smesh), comm);
}

} // namespace mach

#endif 
