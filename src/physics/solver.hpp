#ifndef MACH_SOLVER
#define MACH_SOLVER

#include <fstream>
#include <iostream>
#include <functional>
#include <unordered_map>

#include "adept.h"
#include "json.hpp"
#include "mfem.hpp"

#include "mach_types.hpp"
#include "utils.hpp"
#include "mach_integrator.hpp"
#include "mach_input.hpp"

#ifdef MFEM_USE_PUMI
namespace apf
{
class Mesh2;
}  // namespace apf
#include "PCU.h"
#ifdef MFEM_USE_SIMMETRIX
#include "SimUtil.h"
#include "gmi_sim.h"
#endif  // MFEM_USE_SIMMETRIX

#ifdef MFEM_USE_EGADS
#include "gmi_egads.h"
#endif  // MFEM_USE_EGADS
namespace mach
{
struct pumiDeleter
{
   void operator()(apf::Mesh2 *mesh) const
   {
      mesh->destroyNative();
      apf::destroyMesh(mesh);
   }
};

}  // namespace mach
#endif

namespace mach
{
class MachEvolver;
class MachLoad;

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

   /// Construct the finite element space and perform set-up of derived classes
   /// using virtual functions
   virtual void initDerived();

   /// class destructor
   virtual ~AbstractSolver();

   /// TODO: should this be pretected/private?
   /// Constructs the mesh member based on c preprocesor defs
   /// \param[in] smesh - if provided, defines the mesh for the problem
   void constructMesh(std::unique_ptr<mfem::Mesh> smesh);

   /// Initializes the state vector to a given field.
   /// \param[in] u_init - field that defines the initial condition
   inline virtual void setInitialCondition(const mfem::ParGridFunction &u_init)
   {
      setInitialCondition(*u, u_init);
   }

   /// Initializes the state vector to a given scalar function.
   /// \param[in] u_init - function that defines the initial condition
   inline virtual void setInitialCondition(
       const std::function<double(const mfem::Vector &)> &u_init)
   {
      setInitialCondition(*u, u_init);
   }

   /// Initializes the state vector to a given function.
   /// \param[in] u_init - function that defines the initial condition
   /// \note The second argument in the function `u_init` is the initial
   /// condition value.  This may be a vector of length 1 for scalar.
   inline virtual void setInitialCondition(
       const std::function<void(const mfem::Vector &, mfem::Vector &)> &u_init)
   {
      setInitialCondition(*u, u_init);
   }

   /// Initializes the state variable to a given constant
   /// \param[in] u_init - value that defines the initial condition
   inline virtual void setInitialCondition(const double u_init)
   {
      setInitialCondition(*u, u_init);
   }

   /// Initializes the state variable to a given constant vector
   /// \param[in] u_init - vector that defines the initial condition
   inline virtual void setInitialCondition(const mfem::Vector &u_init)
   {
      setInitialCondition(*u, u_init);
   }

   /// Initializes the state vector to a given field.
   /// \param[in] state - the state vector to initialize
   /// \param[in] u_init - field that defines the initial condition
   inline virtual void setInitialCondition(mfem::ParGridFunction &state,
                                           const mfem::ParGridFunction &u_init)
   {
      state = u_init;
   }

   /// Initializes the state vector to a given scalar function.
   /// \param[in] state - the state vector to initialize
   /// \param[in] u_init - function that defines the initial condition
   virtual void setInitialCondition(
       mfem::ParGridFunction &state,
       const std::function<double(const mfem::Vector &)> &u_init);

   /// Initializes the state vector to a given function.
   /// \param[in] state - the state vector to initialize
   /// \param[in] u_init - function that defines the initial condition
   /// \note The second argument in the function `u_init` is the initial
   /// condition value.  This may be a vector of length 1 for scalar.
   virtual void setInitialCondition(
       mfem::ParGridFunction &state,
       const std::function<void(const mfem::Vector &, mfem::Vector &)> &u_init);

   /// Initializes the state variable to a given constant
   /// \param[in] state - the state vector to initialize
   /// \param[in] u_init - value that defines the initial condition
   virtual void setInitialCondition(mfem::ParGridFunction &state,
                                    const double u_init);

   /// Initializes the state variable to a given constant vector
   /// \param[in] state - the state vector to initialize
   /// \param[in] u_init - vector that defines the initial condition
   virtual void setInitialCondition(mfem::ParGridFunction &state,
                                    const mfem::Vector &u_init);

   /// Initializes the field to a given constant
   /// \param[in] field - the field to set
   /// \param[in] u_init - constant to set the field to
   virtual void setFieldValue(mfem::HypreParVector &field, const double u_init);

   /// Initializes the field to a given constant
   /// \param[in] field_buffer - the field to set
   /// \param[in] u_init - constant to set the field to
   virtual void setFieldValue(double *field_buffer, const double u_init);

   /// Initializes the field to a given scalar function
   /// \param[in] field - the field to set
   /// \param[in] u_init - function that defines spatially varying field
   virtual void setFieldValue(
       mfem::HypreParVector &field,
       const std::function<double(const mfem::Vector &)> &u_init);

   /// Initializes the field to a given scalar function
   /// \param[in] field_buffer - the field to set
   /// \param[in] u_init - function that defines spatially varying field
   virtual void setFieldValue(
       double *field_buffer,
       const std::function<double(const mfem::Vector &)> &u_init);

   /// Initializes the vector field to a given constant vector
   /// \param[in] field - the vector field to set
   /// \param[in] u_init - vector to set the field to
   virtual void setFieldValue(mfem::HypreParVector &field,
                              const mfem::Vector &u_init);

   /// Initializes the vector field to a given constant vector
   /// \param[in] field_buffer - the field to set
   /// \param[in] u_init - vector to set the field to
   virtual void setFieldValue(double *field_buffer, const mfem::Vector &u_init);

   /// Sets the vector field to a given vector-valued function.
   /// \param[in] field - the vector field to set
   /// \param[in] u_init - function that defines spatially varying vector field
   virtual void setFieldValue(
       mfem::HypreParVector &field,
       const std::function<void(const mfem::Vector &, mfem::Vector &)> &u_init);

   /// Sets the vector field to a given vector-valued function.
   /// \param[in] field_buffer - the field to set
   /// \param[in] u_init - function that defines spatially varying vector field
   virtual void setFieldValue(
       double *field_buffer,
       const std::function<void(const mfem::Vector &, mfem::Vector &)> &u_init);

   /// TODO move to protected?
   /// Returns the integral inner product between two grid functions
   /// \param[in] x - grid function
   /// \param[in] y - grid function
   /// \return integral inner product between `x` and `y`
   double calcInnerProduct(const GridFunType &x, const GridFunType &y) const;

   /// Returns the L2 error between the state `u` and given exact solution.
   /// \param[in] u_exact - function that defines the exact solution
   /// \return L2 error
   double calcL2Error(
       const std::function<double(const mfem::Vector &)> &u_exact);

   /// Returns the L2 error between the state `u` and given exact solution.
   /// \param[in] u_exact - function that defines the exact solution
   /// \param[in] entry - if >= 0, the L2 error of state `entry` is returned
   /// \return L2 error
   double calcL2Error(
       const std::function<void(const mfem::Vector &, mfem::Vector &)> &u_exact,
       int entry = -1);

   /// Returns the L2 error of a field and given exact solution.
   /// \param[in] field - grid function to compute L2 error for
   /// \param[in] u_exact - function that defines the exact solution
   /// \param[in] entry - if >= 0, the L2 error of state `entry` is returned
   /// \return L2 error
   double calcL2Error(
       GridFunType *field,
       const std::function<double(const mfem::Vector &)> &u_exact);

   /// Returns the L2 error of a field and given exact solution.
   /// \param[in] field - grid function to compute L2 error for
   /// \param[in] u_exact - function that defines the exact solution
   /// \param[in] entry - if >= 0, the L2 error of state `entry` is returned
   /// \return L2 error
   double calcL2Error(
       GridFunType *field,
       const std::function<void(const mfem::Vector &, mfem::Vector &)> &u_exact,
       int entry = -1);

   /// Returns the L2 error of a field and given exact solution.
   /// \param[in] field - state vector to compute L2 error for
   /// \param[in] u_exact - function that defines the exact solution
   /// \param[in] entry - if >= 0, the L2 error of state `entry` is returned
   /// \return L2 error
   double calcL2Error(
       mfem::HypreParVector &field,
       const std::function<double(const mfem::Vector &)> &u_exact);

   /// Returns the L2 error of a field and given exact solution.
   /// \param[in] field - state vector to compute L2 error for
   /// \param[in] u_exact - function that defines the exact solution
   /// \param[in] entry - if >= 0, the L2 error of state `entry` is returned
   /// \return L2 error
   double calcL2Error(
       mfem::HypreParVector &field,
       const std::function<void(const mfem::Vector &, mfem::Vector &)> &u_exact,
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
   virtual double calcStepSize(int iter,
                               double t,
                               double t_final,
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

   /// Write the mesh and a field to a vtk file
   /// \param[in] file_name - prefix file name **without** .vtk extension
   /// \param[in] field_name - name of field to print
   /// \param[in] refine - if >=0, indicates the number of refinements to make
   /// \param[in] cycle - cycle to write to vtk file
   /// \note the `refine` argument is useful for high-order meshes and
   /// solutions; it divides the elements up so it is possible to visualize.
   void printField(const std::string &file_name,
                   const std::string &field_name,
                   int refine = -1,
                   int cycle = 0)
   {
      auto &field = res_fields.at(field_name);
      printFields(file_name, {&field}, {field_name}, refine, cycle);
   }

   /// Write the mesh and an initializer list to a vtk file
   /// \param[in] file_name - prefix file name **without** .vtk extension
   /// \param[in] field - grid function to print
   /// \param[in] name - name to use for the printed grid function
   /// \param[in] refine - if >=0, indicates the number of refinements to make
   /// \param[in] cycle - cycle to write to vtk file
   /// \note the `refine` argument is useful for high-order meshes and
   /// solutions; it divides the elements up so it is possible to visualize.
   void printField(const std::string &file_name,
                   mfem::ParGridFunction &field,
                   const std::string &name,
                   int refine = -1,
                   int cycle = 0)
   {
      printFields(file_name, {&field}, {name}, refine, cycle);
   }

   /// Write the mesh and an initializer list to a vtk file
   /// \param[in] file_name - prefix file name **without** .vtk extension
   /// \param[in] fields - list of grid functions to print, passed as an
   ///                     initializer list
   /// \param[in] names - list of names to use for each grid function printed
   /// \param[in] refine - if >=0, indicates the number of refinements to make
   /// \param[in] cycle - cycle to write to vtk file
   /// \note the `refine` argument is useful for high-order meshes and
   /// solutions; it divides the elements up so it is possible to visualize.
   void printFields(const std::string &file_name,
                    std::vector<mfem::ParGridFunction *> fields,
                    std::vector<std::string> names,
                    int refine = -1,
                    int cycle = 0);

   /// \brief Returns a vector of pointers to grid functions that define fields
   /// Default behavior is to return just the state `u`
   virtual std::vector<GridFunType *> getFields();

   // mfem::ParGridFunction& getField(std::string field)
   // { return res_fields.at(field); }
   void getField(const std::string &name, double *field_buffer);

   void getField(const std::string &name, mfem::HypreParVector &field);

   /// DEPRECIATED -> use version with HypreParVector
   /// Solve for the state variables based on current mesh, solver, etc.
   virtual void solveForState() { solveForState(*u); }

   /// DEPRECIATED -> use version with HypreParVector
   virtual void solveForState(mfem::ParGridFunction &state);

   inline virtual void solveForState(double *state)
   {
      MachInputs inputs;
      solveForState(inputs, state);
   }

   inline virtual void solveForState(mfem::HypreParVector &state)
   {
      MachInputs inputs;
      solveForState(inputs, state);
   }

   virtual void solveForState(const MachInputs &inputs, double *state);

   virtual void solveForState(const MachInputs &inputs,
                              mfem::HypreParVector &state);

   /// Solve for the adjoint based on current mesh, solver, etc.
   /// \param[in] fun - specifies the functional corresponding to the adjoint
   virtual void solveForAdjoint(const std::string &fun);

   /// Solve for the adjoint with a given right hand side (instead of )
   virtual void solveForAdjoint(const mfem::Vector &rhs) { }

   /// Check the Jacobian using a finite-difference directional derivative
   /// \param[in] pert - function that defines the perturbation direction
   /// \note Compare the results of the project Jac*pert using the Jacobian
   /// directly versus a finite-difference based product.
   void checkJacobian(const mfem::ParGridFunction &state,
                      std::function<double(const mfem::Vector &)> pert_fun);

   /// Check the Jacobian using a finite-difference directional derivative
   /// \param[in] pert - function that defines the perturbation direction
   /// \note Compare the results of the project Jac*pert using the Jacobian
   /// directly versus a finite-difference based product.
   void checkJacobian(void (*pert_fun)(const mfem::Vector &, mfem::Vector &));

   /// Creates the nonlinear form for the functional
   /// \param[in] fun - specifies the desired functional
   /// \note if a nonlinear form for `fun` has already been created an
   /// exception will be thrown
   void createOutput(const std::string &fun);

   /// Creates the nonlinear form for the functional
   /// \param[in] fun - specifies the desired functional
   /// \param[in] options - options needed for calculating functional
   /// \note if a nonlinear form for `fun` has already been created an
   /// exception will be thrown
   void createOutput(const std::string &fun, const nlohmann::json &options);

   /// Evaluate and return the output functional specified by `fun`
   /// \param[in] fun - specifies the desired functional
   /// \returns scalar value of estimated functional value
   double calcOutput(const std::string &fun) { return calcOutput(*u, fun); }

   /// Evaluate and return the output functional specified by `fun`
   /// \param[in] state - the state vector to evaluate the functional at
   /// \param[in] fun - specifies the desired functional
   /// \returns scalar value of estimated functional value
   double calcOutput(const mfem::ParGridFunction &state,
                     const std::string &fun);

   /// Evaluates and returns the output functional specifed by `fun`
   /// \param[in] fun - specifies the desired functional
   /// \param[in] inputs - collection of field or scalar inputs to set before
   ///                     evaluating functional
   /// \return scalar value of estimated functional value
   double calcOutput(const std::string &fun, const MachInputs &inputs);

   /// Evaluates and returns the partial derivative of output functional
   /// specifed by `of` with respect to the input specified by `wrt`
   /// \param[in] of - specifies the desired functional
   /// \param[in] wrt - specifies the input to differentiate with respect to
   /// \param[in] inputs - collection of field or scalar inputs to set before
   ///                     evaluating functional
   /// \param[out] partial - the partial with respect to a scalar-valued input
   void calcOutputPartial(const std::string &of,
                          const std::string &wrt,
                          const MachInputs &inputs,
                          double &partial);

   /// Evaluates and returns the partial derivative of output functional
   /// specifed by `of` with respect to the input specified by `wrt`
   /// \param[in] of - specifies the desired functional
   /// \param[in] wrt - specifies the input to differentiate with respect to
   /// \param[in] inputs - collection of field or scalar inputs to set before
   ///                     evaluating functional
   /// \param[out] partial - the partial with respect to a vector-valued input
   void calcOutputPartial(const std::string &of,
                          const std::string &wrt,
                          const MachInputs &inputs,
                          double *partial);

   /// Evaluates and returns the partial derivative of output functional
   /// specifed by `of` with respect to the input specified by `wrt`
   /// \param[in] of - specifies the desired functional
   /// \param[in] wrt - specifies the input to differentiate with respect to
   /// \param[in] inputs - collection of field or scalar inputs to set before
   ///                     evaluating functional
   /// \param[out] partial - the partial with respect to a vector-valued input
   void calcOutputPartial(const std::string &of,
                          const std::string &wrt,
                          const MachInputs &inputs,
                          mfem::HypreParVector &partial);

   /// Sets options for the output functional specifed by `fun`
   /// \param[in] fun - specifies the desired functional
   /// \param[in] options - options needed for calculating functional
   /// \note will only have an effect if a subclass supports setting options
   ///       for the functional
   virtual void setOutputOptions(const std::string &fun,
                                 const nlohmann::json &options)
   { }

   /// Compute the residual norm based on the current solution in `u`
   /// \returns the l2 (discrete) norm of the residual evaluated at `u`
   double calcResidualNorm() const { return calcResidualNorm(*u); }

   /// Compute the residual norm based on the input `state`
   /// \returns the l2 (discrete) norm of the residual evaluated at `u`
   double calcResidualNorm(const mfem::ParGridFunction &state) const;

   /// Return a state sized vector constructed from an externally allocated
   /// array \param[in] data - external data array \note If `data` is nullptr a
   /// new array will be allocated. If `data` is not `nullptr` it is assumed to
   /// be of size of at least `fes->GetVSize()`
   // std::unique_ptr<mfem::ParGridFunction> getNewField(double *data =
   // nullptr);

   /// Return a state sized vector constructed from an externally allocated
   /// array \param[in] data - external data array \note If `data` is nullptr a
   /// new array will be allocated. If `data` is not `nullptr` it is assumed to
   /// be of size of at least `fes->GetTrueVSize()`
   std::unique_ptr<mfem::HypreParVector> getNewField(double *data = nullptr);

   /// Compute the residual based on the current solution in `u`
   /// \param[out] residual - the residual
   void calcResidual(mfem::ParGridFunction &residual) const
   {
      calcResidual(*u, residual);
   }

   /// Compute the residual based on `state` and store the it in `residual`
   /// \param[in] state - the current state to evaluate the residual at
   /// \param[out] residual - the residual
   virtual void calcResidual(const mfem::ParGridFunction &state,
                             mfem::ParGridFunction &residual) const;

   /// Compute the residual based on inputs and store the it in `residual`
   /// \param[in] inputs - collection of field or scalar inputs to set before
   ///                     evaluating residual
   /// \param[out] residual - the residual
   void calcResidual(const MachInputs &inputs, double *res_buffer) const;

   /// Compute the residual based on inputs and store the it in `residual`
   /// \param[in] inputs - collection of field or scalar inputs to set before
   ///                     evaluating residual
   /// \param[out] residual - the residual
   void calcResidual(const MachInputs &inputs,
                     mfem::HypreParVector &residual) const;

   /// Set inputs for residual integrators and assemble state jacobian
   void linearize(const MachInputs &inputs);

   /// Compute vector jacobian product for derivative with respect to a scalar
   /// \param[in] residual_bar - multiplies jacobian on the left hand side
   /// \param[in] wrt - string identifying what the jacobian is taken with
   /// respect to \return result of vector jacobian product
   double vectorJacobianProduct(double *res_bar_buffer, const std::string &wrt);

   /// Compute vector jacobian product for derivative with respect to a scalar
   /// \param[in] residual_bar - multiplies jacobian on the left hand side
   /// \param[in] wrt - string identifying what the jacobian is taken with
   /// respect to \return result of vector jacobian product
   double vectorJacobianProduct(const mfem::HypreParVector &res_bar,
                                const std::string &wrt);

   /// Compute vector jacobian product for derivative with respect to a vector
   /// \param[in] residual_bar - multiplies jacobian on the left hand side
   /// \param[in] wrt - string identifying what the jacobian is taken with
   /// respect to \param[inout] wrt_bar - result of vector jacobian product
   /// added to wrt_bar
   void vectorJacobianProduct(double *res_bar_buffer,
                              const std::string &wrt,
                              double *wrt_bar);

   /// Compute vector jacobian product for derivative with respect to a vector
   /// \param[in] residual_bar - multiplies jacobian on the left hand side
   /// \param[in] wrt - string identifying what the jacobian is taken with
   /// respect to \param[inout] wrt_bar - result of vector jacobian product
   /// added to wrt_bar
   void vectorJacobianProduct(const mfem::HypreParVector &res_bar,
                              const std::string &wrt,
                              mfem::HypreParVector &wrt_bar);

   /// TODO: Who added this?  Do we need it still?  What is it for?  Document!
   void feedpert(void (*p)(const mfem::Vector &, mfem::Vector &)) { pert = p; }

   /// Return the output map
   std::map<std::string, NonlinearFormType> GetOutput() const { return output; }

   /// convert conservative variables to entropy variables
   /// \param[in/out] state - the conservative/entropy variables
   // virtual void convertToEntvar(mfem::Vector &state) { }

   /// Compute the sensitivity of an output to the mesh nodes, using appropriate
   /// mesh sensitivity integrators. Need to compute the adjoint first.
   virtual mfem::Vector *getMeshSensitivities();

   /// Return a pointer to the solver's mesh
   MeshType *getMesh() { return mesh.get(); }

   void printMesh(const std::string &filename)
   {
      mesh->PrintVTU(filename, mfem::VTKFormat::BINARY, true, 0);
   }
   /// return a reference to the mesh's coordinate field
   mfem::GridFunction &getMeshCoordinates() { return *mesh->GetNodes(); }

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
   inline int getFieldSize(std::string field)
   {
      return res_fields.at(field).ParFESpace()->GetTrueVSize();
   }

   /// return the options dictionary with read-only access
   inline const nlohmann::json &getOptions() const { return options; }

#ifdef MFEM_USE_PUMI
   /// Return a pointer to the underlying PUMI mesh
   apf::Mesh2 *getPumiMesh() { return pumi_mesh.get(); }
#endif

   /// Tell the underling forms that the mesh has changed;
   virtual void Update() { fes->Update(); }

   // /// Set the data for the input field
   // /// \param[in] name - name of the field
   // /// \param[in] field - reference the existing field
   // /// \note it is assumed that this external grid function is defined on the
   // /// same mesh the solver uses
   // void setResidualInput(std::string name,
   //                       mfem::ParGridFunction &field);

   // /// Set the data for the input field
   // /// \param[in] name - name of the field
   // /// \param[in] field - data buffer for an external grid function
   // /// \note it is assumed that this external grid function is defined on the
   // /// same mesh the solver uses
   // void setResidualInput(std::string name,
   //                       double *field);

   // /// Compute seed^T \frac{\partial R}{\partial field}
   // /// \param[in] field - name of the field to differentiate with respect to
   // /// \param[in] seed - the field to contract with (usually the adjoint)
   // mfem::HypreParVector* vectorJacobianProduct(std::string field,
   //                                             mfem::ParGridFunction &seed);

   // /// Register a functional's dependence on a field
   // /// \param[in] fun - specifies the desired functional
   // /// \param[in] name - name of the field
   // /// \param[in] field - reference the existing field
   // /// \note field/name pairs are stored in `external_fields`
   // void setFunctionalInput(std::string fun,
   //                         std::string name,
   //                         mfem::ParGridFunction &field);

   // /// Compute \frac{\partial J}{\partial field}
   // /// \param[in] fun - specifies the desired functional
   // /// \param[in] field - name of the field to differentiate with respect to
   // mfem::HypreParVector* calcFunctionalGradient(std::string fun,
   //                                              std::string field);

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
   bool PCU_previously_initialized = false;
#endif

   //--------------------------------------------------------------------------
   // Members associated with fields
   /// finite element or SBP operators
   std::unique_ptr<mfem::FiniteElementCollection> fec;
   /// discrete finite element space
   std::unique_ptr<SpaceType> fes;
   /// pointer to mesh's underlying finite element space
   SpaceType *mesh_fes;
   /// state sized work vector
   std::unique_ptr<mfem::ParGridFunction> scratch;
   /// state tdof sized work vector
   mutable std::unique_ptr<mfem::HypreParVector> scratch_tv;
   /// state variable
   std::unique_ptr<GridFunType> u;
   /// initial state variable
   std::unique_ptr<GridFunType> u_init;
   /// prior state variable
   std::unique_ptr<GridFunType> u_old;
   /// time derivative at current step
   std::unique_ptr<GridFunType> dudt;
   /// adjoint variable
   std::unique_ptr<GridFunType> adj;
   /// prior adjoint variable (forward in time)
   std::unique_ptr<GridFunType> adj_old;
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
   std::unique_ptr<MachLoad> load;
   /// entropy/energy that is needed for RRK methods
   std::unique_ptr<NonlinearFormType> ent;

   //--------------------------------------------------------------------------
   // Members associated with external inputs
   /// map of external fields the residual depends on
   std::unordered_map<std::string, mfem::ParGridFunction> res_fields;
   /// map of external true vectors, used primarily for accumulating bar
   /// variables
   std::map<std::string, mfem::HypreParVector> ext_tvs;
   /// collection of integrators for the residual
   std::vector<MachIntegrator> res_integrators;
   /// map of linear forms that will compute
   /// \psi^T \frac{\partial R}{\partial field}
   /// for each field the residual depends on
   std::map<std::string, mfem::ParLinearForm> res_sens;
   /// map of nonlinear forms that will compute
   /// \psi^T \frac{\partial R}{\partial scalar}
   /// for each scalar the residual depends on
   std::map<std::string, mfem::ParNonlinearForm> res_scalar_sens;

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
   /// Array that hold mesh fes degrees of freedom on model surfaces
   mfem::Array<int> mesh_fes_surface_dofs;
   /// Array that holds fes degrees of freedom on model surfaces
   mfem::Array<int> fes_surface_dofs;
   /// `bndry_marker[i]` lists the boundaries associated with a particular BC
   std::vector<mfem::Array<int>> bndry_marker;
   /// map of output functionals
   std::map<std::string, NonlinearFormType> output;
   /// collection of integrators for each functional
   std::map<std::string, std::vector<MachIntegrator>> fun_integrators;
   /// map of linear forms that will compute \frac{\partial J}{\partial field}
   /// for each field the functional depends on
   std::map<std::string, std::map<std::string, mfem::ParLinearForm>>
       output_sens;
   /// map of nonlinear forms that will compute
   /// \frac{\partial J}{\partial scalar} for each scalar the functional
   /// depends on
   std::map<std::string, std::map<std::string, mfem::ParNonlinearForm>>
       output_scalar_sens;

   /// map of fractional functionals - a funtional that is a fraction of others
   std::unordered_map<std::string, std::vector<std::string>> fractional_output;
   /// output_bndry_marker[fun] lists the boundaries associated with output fun
   std::unordered_map<std::string, mfem::Array<int>> output_bndry_marker;

   //--------------------------------------------------------------------------

   /// Construct PUMI Mesh
   void constructPumiMesh();

   /// Remove internal mesh boundaries based on options
   void removeInternalBoundaries();

   void setUpExternalFields();

   /// Construct various coefficients
   virtual void constructCoefficients() { }

   /// Initialize all forms needed by the derived class
   /// \note Derived classes must allocate the forms they need to use.  Only
   /// allocated forms will have integrators added to them.
   virtual void constructForms() = 0;

   /// Add domain integrators to `mass`
   /// \param[in] alpha - scales the data; used to move terms to rhs or lhs
   virtual void addMassIntegrators(double alpha);

   /// Add domain integrators to `nonlinear_mass`
   /// \param[in] alpha - scales the data; used to move terms to rhs or lhs
   virtual void addNonlinearMassIntegrators(double alpha) { }

   /// Add volume integrators to `res` based on `options`
   /// \param[in] alpha - scales the data; used to move terms to rhs or lhs
   virtual void addResVolumeIntegrators(double alpha) { }

   /// Add boundary-face integrators to `res` based on `options`
   /// \param[in] alpha - scales the data; used to move terms to rhs or lhs
   virtual void addResBoundaryIntegrators(double alpha) { }

   /// Add interior-face integrators to `res` based on `options`
   /// \param[in] alpha - scales the data; used to move terms to rhs or lhs
   virtual void addResInterfaceIntegrators(double alpha) { }

   /// Add volume integrators to `stiff`
   /// \param[in] alpha - scales the data; used to move terms to rhs or lhs
   virtual void addStiffVolumeIntegrators(double alpha) { }

   /// Add boundary-face integrators to `stiff`
   /// \param[in] alpha - scales the data; used to move terms to rhs or lhs
   virtual void addStiffBoundaryIntegrators(double alpha) { }

   /// Add interior-face integrators to `stiff`
   /// \param[in] alpha - scales the data; used to move terms to rhs or lhs
   virtual void addStiffInterfaceIntegrators(double alpha) { }

   /// Add volume integrators to 'load'
   /// \param[in] alpha - scales the data; used to move terms to rhs or lhs
   virtual void addLoadVolumeIntegrators(double alpha) { }

   /// Add boundary-face integrators to `load'
   /// \param[in] alpha - scales the data; used to move terms to rhs or lhs
   virtual void addLoadBoundaryIntegrators(double alpha) { }

   /// Add interior-face integrators to `load'
   /// \param[in] alpha - scales the data; used to move terms to rhs or lhs
   virtual void addLoadInterfaceIntegrators(double alpha) { }

   /// Construct load vector
   /// \param[in] alpha - scales the data; used to move terms to rhs or lhs
   /// \note - only implement this method if `load` is a GridFunction, and not
   /// a LinearForm
   virtual void assembleLoadVector(double alpha) { }

   /// Add volume integrators for `ent`
   virtual void addEntVolumeIntegrators() { }

   /// mark which boundaries are essential
   virtual void setEssentialBoundaries();

   /// Define the number of states, the finite element space, and state u
   virtual int getNumState() = 0;

   /// Add integrators to functional `fun` based on options
   virtual void addOutputIntegrators(const std::string &fun,
                                     const nlohmann::json &options)
   { }

   /// Solve for the steady state problem using newton method
   virtual void solveSteady(mfem::ParGridFunction &state);

   /// Solve for a transient state using a selected time-marching scheme
   virtual void solveUnsteady(mfem::ParGridFunction &state);

   /// For code that should be executed before the time stepping begins
   /// \param[in] state - the current state
   virtual void initialHook(const mfem::ParGridFunction &state) { }

   /// For code that should be executed before `ode_solver->Step`
   /// \param[in] iter - the current iteration
   /// \param[in] t - the current time (before the step)
   /// \param[in] dt - the step size that will be taken
   /// \param[in] state - the current state
   virtual void iterationHook(int iter,
                              double t,
                              double dt,
                              const mfem::ParGridFunction &state)
   { }

   /// Determines when to exit the time stepping loop
   /// \param[in] iter - the current iteration
   /// \param[in] t - the current time (after the step)
   /// \param[in] t_final - the final time
   /// \param[in] dt - the step size that was just taken
   /// \param[in] state - the current state
   virtual bool iterationExit(int iter,
                              double t,
                              double t_final,
                              double dt,
                              const mfem::ParGridFunction &state) const;

   /// For code that should be executed after the time stepping ends
   /// \param[in] iter - the terminal iteration
   /// \param[in] t_final - the final time
   /// \param[in] state - the current state
   virtual void terminalHook(int iter,
                             double t_final,
                             const mfem::ParGridFunction &state)
   { }

   /// Solve for a steady adjoint
   /// \param[in] fun - specifies the functional corresponding to the adjoint
   virtual void solveSteadyAdjoint(const std::string &fun);

   /// Solve for an unsteady adjoint
   /// \param[in] fun - specifies the functional corresponding to the adjoint
   virtual void solveUnsteadyAdjoint(const std::string &fun);

   /// TODO: What is this doing here?
   void (*pert)(const mfem::Vector &, mfem::Vector &){};

   /// Construct a preconditioner based on the given options
   /// \param[in] options - options structure that determines preconditioner
   /// \returns unique pointer to the preconditioner object
   virtual std::unique_ptr<mfem::Solver> constructPreconditioner(
       nlohmann::json &options);

   /// Constuct a linear system solver based on the given options
   /// \param[in] options - options structure that determines the solver
   /// \param[in] prec - preconditioner object for iterative solvers
   /// \returns unique pointer to the linear solver object
   virtual std::unique_ptr<mfem::Solver> constructLinearSolver(
       nlohmann::json &options,
       mfem::Solver &prec);

   /// Constructs the nonlinear solver object
   /// \param[in] options - options structure that determines the solver
   /// \param[in] lin_solver - linear solver for the Newton steps
   /// \returns unique pointer to the Newton solver object
   virtual std::unique_ptr<mfem::NewtonSolver> constructNonlinearSolver(
       nlohmann::json &options,
       mfem::Solver &lin_solver);

   /// Constructs the operator that defines ODE evolution
   virtual void constructEvolver();

   /// Used by derived classes that themselves construct solver objects that
   /// don't need all the memory for a fully featured solver, that just need to
   /// support the AbstractSolver interface (JouleSolver)
   AbstractSolver(const std::string &opt_file_name,
                  MPI_Comm comm = MPI_COMM_WORLD);

   /// calculate a functional that is the product of others
   /// \param[in] state - the state vector to evaluate the functional at
   /// \param[in] fun - specifies the desired functional
   /// \returns scalar value of estimated functional value
   double calcFractionalOutput(const mfem::ParGridFunction &state,
                               const std::string &fun);

   /// Add integrators to the linear form representing the product
   /// seed^T \frac{\partial R}{\partial field} for a particular field
   /// \param[in] name - name of the field for the integrators
   /// \param[in] seed - the field to contract with (usually the adjoint)
   virtual void addResFieldSensIntegrators(std::string field,
                                           mfem::ParGridFunction &seed)
   { }

   /// Add integrators to the linear form representing the vector
   /// \frac{\partial J}{\partial field} for a particular field
   /// \param[in] fun - specifies the desired functional
   /// \param[in] name - name of the field for the integrators
   virtual void addFuncFieldSensIntegrators(std::string fun, std::string field)
   { }

   /// Iterates through each input and calls `setInput` for each
   /// \param[in] integrators - list of integrators to set scalar inputs for
   /// \param[in] inputs - collection of named field or scalar inputs
   void setInputs(std::vector<MachIntegrator> &integrators,
                  const MachInputs &inputs);

   /// If the input is a field variable, updates the data for the field in
   /// `res_fields`. If the input is a scalar, iterates through the integrators
   /// and calls `setInput` for each integrator to set it's scalar inputs
   /// \param[in] integrators - list of integrators to set scalar inputs for
   /// \param[in] name - name of input
   /// \param[in] input - input to set, either a field or scalar
   void setInput(std::vector<MachIntegrator> &integrators,
                 const std::string &name,
                 const MachInput &input);

   /// Adds domain integrator to the nonlinear form for `fun`, and adds
   /// reference to it to in fun_integrators as a MachIntegrator
   /// \param[in] fun - specifies the desired functional
   /// \param[in] integrator - integrator to add to functional
   /// \tparam T - type of integrator, used for constructing MachIntegrator
   template <typename T>
   void addResidualDomainIntegrator(T *integrator)
   {
      res->AddDomainIntegrator(integrator);
      res_integrators.emplace_back(*integrator);
      mach::addSensitivityIntegrator(
          *integrator, res_fields, res_sens, res_scalar_sens);
   }

   /// Adds interface integrator to the nonlinear form for `fun`, and adds
   /// reference to it to in fun_integrators as a MachIntegrator
   /// \param[in] fun - specifies the desired functional
   /// \param[in] integrator - integrator to add to functional
   /// \tparam T - type of integrator, used for constructing MachIntegrator
   template <typename T>
   void addResidualInteriorFaceIntegrator(T *integrator)
   {
      res->AddInteriorFaceIntegrator(integrator);
      res_integrators.emplace_back(*integrator);
      mach::addSensitivityIntegrator(
          *integrator, res_fields, res_sens, res_scalar_sens);
   }

   /// Adds boundary integrator to the nonlinear form for `fun`, and adds
   /// reference to it to in fun_integrators as a MachIntegrator
   /// \param[in] fun - specifies the desired functional
   /// \param[in] integrator - integrator to add to functional
   /// \tparam T - type of integrator, used for constructing MachIntegrator
   template <typename T>
   void addResidualBdrFaceIntegrator(T *integrator,
                                     mfem::Array<int> &bdr_marker)
   {
      res->AddBdrFaceIntegrator(integrator, bdr_marker);
      res_integrators.emplace_back(*integrator);
      mach::addSensitivityIntegrator(
          *integrator, res_fields, res_sens, res_scalar_sens);
   }

   /// Adds domain integrator to the nonlinear form for `fun`, and adds
   /// reference to it to in fun_integrators as a MachIntegrator
   /// \param[in] fun - specifies the desired functional
   /// \param[in] integrator - integrator to add to functional
   /// \tparam T - type of integrator, used for constructing MachIntegrator
   template <typename T>
   void addOutputDomainIntegrator(const std::string &fun, T *integrator)
   {
      output.at(fun).AddDomainIntegrator(integrator);
      fun_integrators.at(fun).emplace_back(*integrator);
      mach::addSensitivityIntegrator(
          *integrator, res_fields, output_sens[fun], output_scalar_sens[fun]);
   }

   /// Adds interface integrator to the nonlinear form for `fun`, and adds
   /// reference to it to in fun_integrators as a MachIntegrator
   /// \param[in] fun - specifies the desired functional
   /// \param[in] integrator - integrator to add to functional
   /// \tparam T - type of integrator, used for constructing MachIntegrator
   template <typename T>
   void addOutputInteriorFaceIntegrator(const std::string &fun, T *integrator)
   {
      output.at(fun).AddInteriorFaceIntegrator(integrator);
      fun_integrators.at(fun).emplace_back(*integrator);
      mach::addSensitivityIntegrator(
          *integrator, res_fields, output_sens[fun], output_scalar_sens[fun]);
   }

   /// Adds boundary integrator to the nonlinear form for `fun`, and adds
   /// reference to it to in fun_integrators as a MachIntegrator
   /// \param[in] fun - specifies the desired functional
   /// \param[in] integrator - integrator to add to functional
   /// \tparam T - type of integrator, used for constructing MachIntegrator
   template <typename T>
   void addOutputBdrFaceIntegrator(const std::string &fun,
                                   T *integrator,
                                   mfem::Array<int> &bdr_marker)
   {
      output.at(fun).AddBdrFaceIntegrator(integrator, bdr_marker);
      fun_integrators.at(fun).emplace_back(*integrator);
      mach::addSensitivityIntegrator(
          *integrator, res_fields, output_sens[fun], output_scalar_sens[fun]);
   }

private:
   /// explicitly prohibit copy construction
   AbstractSolver(const AbstractSolver &) = delete;
   AbstractSolver &operator=(const AbstractSolver &) = delete;

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
   // auto solver = std::make_unique<DerivedSolver>(opt_file_name, move(smesh));
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

}  // namespace mach

#endif
