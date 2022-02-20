#ifndef MACH_SOLVER
#define MACH_SOLVER

#include <fstream>
#include <iostream>

#include "adept.h"
#include "mfem.hpp"
#include "centgridfunc.hpp"
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
   /// \param[in] smesh - if provided, defines the mesh for the problem
   AbstractSolver(const std::string &opt_file_name =
                  std::string("mach_options.json"),
                  std::unique_ptr<mfem::Mesh> smesh = nullptr);

   /// Perform set-up of derived classes using virtual functions
   /// \todo Put the constructors and this in a factory
   void initDerived();
   void initDerived(mfem::Array<mfem::Vector *> &center);

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

   void setMinL2ErrorInitialCondition(void (*u_init)(const mfem::Vector &,
                                           mfem::Vector &)); 

   void setInverseInitialCondition(void (*u_init)(const mfem::Vector &,
                                           mfem::Vector &));

   /// Initializes the state variable to a given constant
   /// \param[in] u_init - vector that defines the initial condition
   void setInitialCondition(const mfem::Vector &uic); 

   /// Returns the integral inner product between two grid functions
   /// \param[in] x - grid function 
   /// \param[in] y - grid function 
   /// \returns integral inner product between `x` and `y`
   double calcInnerProduct(const GridFunType &x, const GridFunType &y);

   /// Returns the L2 error between the state `u` and given exact solution.
   /// \param[in] u_exact - function that defines the exact solution
   /// \param[in] entry - if >= 0, the L2 error of state `entry` is returned
   /// \returns L2 error
   double calcL2Error(void (*u_exact)(const mfem::Vector &, mfem::Vector &),
                      int entry = -1);
   double calcSodShockL1Error(void (*u_exact)(const mfem::Vector &, mfem::Vector &),
                      int entry = -1);
   double calcSodShockMaxError(void (*u_exact)(const mfem::Vector &, mfem::Vector &),
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
   void printResidual(const std::string &file_name);

   /// Solve for the state variables based on current mesh, solver, etc.
   void solveForState();
   
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
   double calcOutput(const std::string &fun);
   
   /// Compute the residual norm based on the current solution in `u`
   /// \returns the l2 (discrete) norm of the residual evaluated at `u`
   double calcResidualNorm();

   void feedpert(void (*p)(const mfem::Vector &, mfem::Vector &))
   {
      pert = p;
   }

   void printError(const std::string &file_name, int refine,
                     void (*u_exact)(const mfem::Vector &, mfem::Vector &));

   /// return the out put map
   std::map<std::string, NonlinearFormType> GetOutput()
   {
      return output;
   }

   /// A temporal funtion that print the 2d sod_shock problem
   virtual void PrintSodShock(const std::string &file_name) = 0;
   virtual void PrintSodShockCenter(const std::string &file_name) = 0;
   
   /// A virtual function convert the conservative variable to entropy variables
   /// defined in EulerSolver
   /// \param[in/out] state - state vector
   virtual void convertToEntvar(mfem::Vector &state) = 0;
   virtual void convertToConserv(mfem::Vector &state) = 0;
   virtual void convertToConservCent(mfem::Vector &state) = 0;
   virtual void checkConversion(void (*u_exact)(const mfem::Vector &, mfem::Vector &)) = 0;
   virtual void conToEntropyVars(const mfem::Vector &entropy, mfem::Vector &conserv) = 0;
   virtual void conToConservVars(const mfem::Vector &conserv, mfem::Vector &entropy) = 0;
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
   std::unique_ptr<SpaceType> fes_normal; 
   /// state variable
   std::unique_ptr<GridFunType> u;
   std::unique_ptr<mfem::CentGridFunction> uc;
   /// adjoint variable 
   std::unique_ptr<GridFunType> adj;
   /// the spatial residual (a semilinear form)
   std::unique_ptr<NonlinearFormType> res;
   /// time-marching method (might be NULL)
   std::unique_ptr<mfem::ODESolver> ode_solver;
   /// the mass matrix bilinear form
   std::unique_ptr<BilinearFormType> mass;
   /// the nonlinear form evaluate the mass matrix
   std::unique_ptr<NonlinearFormType> nonlinear_mass;
   /// mass matrix
   std::unique_ptr<MatrixType> mass_matrix;
   std::unique_ptr<MatrixType> mass_matrix_gd;
   std::unique_ptr<MatrixType> mass_lump;
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

   /// Add volume integrators to `res` based on `options`
   /// \param[in] alpha - scales the data; used to move terms to rhs or lhs
   virtual void addVolumeIntegrators(double alpha) {};

   /// Add mass integrators to `nonlinear_mass` based on `options`
   /// \param[in] alpha - scales the data; used to move terms to rhs or lhs
   virtual void addMassIntegrators(double alpha) {};

   /// Add boundary-face integrators to `res` based on `options`
   /// \param[in] alpha - scales the data; used to move terms to rhs or lhs
   virtual void addBoundaryIntegrators(double alpha) {};

   /// Add interior-face integrators to `res` based on `options`
   /// \param[in] alpha - scales the data; used to move terms to rhs or lhs
   virtual void addInterfaceIntegrators(double alpha) {};

   /// Define the number of states, the finite element space, and state u
   virtual int getNumState() = 0; 

   /// Create `output` based on `options` and add approporiate integrators
   virtual void addOutputs() {};

   /// Solve for the steady state problem using newton method
   virtual void solveSteady();

   /// Solve for a transient state using a selected time-marching scheme
   virtual void solveUnsteady();
   
   /// Solve for a steady adjoint
   /// \param[in] fun - specifies the functional corresponding to the adjoint
   virtual void solveSteadyAdjoint(const std::string &fun);

   /// Solve for an unsteady adjoint
   /// \param[in] fun - specifies the functional corresponding to the adjoint
   virtual void solveUnsteadyAdjoint(const std::string &fun);

   /// Defined in deerived class that update the nonlinear form mass matrix
   virtual void updateNonlinearMass(int ti, double dt, double alpha) {};

   void (*pert)(const mfem::Vector &, mfem::Vector &);
};

} // namespace mach

#endif 
