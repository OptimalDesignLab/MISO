#ifndef MISO_PDE_SOLVER
#define MISO_PDE_SOLVER

#include <memory>
#include <vector>
#include <functional>

#include "mfem.hpp"
#include "nlohmann/json.hpp"
#ifdef MFEM_USE_PUMI
#include "apf.h"
#include "apfMesh2.h"

namespace miso
{
struct pumiDeleter
{
   void operator()(apf::Mesh2 *mesh) const
   {
      if (mesh != nullptr)
      {
         mesh->destroyNative();
         apf::destroyMesh(mesh);
      }
   }
};

}  // namespace miso
#endif

#include "abstract_solver.hpp"
#include "finite_element_state.hpp"
#include "finite_element_dual.hpp"

namespace miso
{
struct MISOMesh
{
   std::unique_ptr<mfem::ParMesh> mesh = nullptr;
#ifdef MFEM_USE_PUMI
   std::unique_ptr<apf::Mesh2, pumiDeleter> pumi_mesh = nullptr;
   static int pumi_mesh_count;
   static bool PCU_previously_initialized;

   MISOMesh();

   MISOMesh(const MISOMesh &) = delete;
   MISOMesh &operator=(const MISOMesh &) = delete;

   MISOMesh(MISOMesh &&) noexcept;
   MISOMesh &operator=(MISOMesh &&) noexcept;

   ~MISOMesh();
#endif
};

MISOMesh constructMesh(MPI_Comm comm,
                       const nlohmann::json &mesh_options,
                       std::unique_ptr<mfem::Mesh> smesh = nullptr,
                       bool keep_boundaries = false);

MISOMesh constructPumiMesh(MPI_Comm comm, const nlohmann::json &mesh_options);

class PDESolver : public AbstractSolver2
{
public:
   int getNumStates() const { return fes().GetVDim(); }
   int getFieldSize(const std::string &name) const override;
   void getMeshCoordinates(mfem::Vector &mesh_coords) const;

   FiniteElementState &getState() { return state(); }
   const FiniteElementState &getState() const { return state(); }

   FiniteElementDual &getResVec() { return res_vec(); }
   const FiniteElementDual &getResVec() const { return res_vec(); }

   /// Construct a `PDESolver`
   /// \param[in] incomm - MPI communicator to associate with the solver
   /// \param[in] solver_options - options used to define the solver
   /// \param[in] num_states - number of states at each degree of freedom
   /// \param[in] smesh - serial mesh for the solver (optional)
   PDESolver(MPI_Comm incomm,
             const nlohmann::json &solver_options,
             const int num_states,
             std::unique_ptr<mfem::Mesh> smesh = nullptr);

   /// Construct a `PDESolver` whose num of states is determined by a function
   /// \param[in] incomm - MPI communicator to associate with the solver
   /// \param[in] solver_options - options used to define the solver
   /// \param[in] num_states - a function that returns the number of states
   /// \param[in] smesh - serial mesh for the solver (optional)
   /// \note This version is needed when the number of states depends on the
   /// mesh, but the mesh is not available until after it has been loaded.  In
   /// this case, the `num_states` function can be used after the mesh is
   /// available to determine the number of states.
   PDESolver(MPI_Comm incomm,
             const nlohmann::json &solver_options,
             std::function<int(const nlohmann::json &, int)> num_states,
             std::unique_ptr<mfem::Mesh> smesh = nullptr);

protected:
   /// object defining the mfem computational mesh
   MISOMesh mesh_;

   /// Reference to solver state vector
   mfem::ParMesh &mesh() { return *mesh_.mesh; }
   const mfem::ParMesh &mesh() const { return *mesh_.mesh; }

   /// solver material properties
   nlohmann::json materials;

   /// Members associated with fields
   /// Map of all state vectors used by the solver
   std::map<std::string, FiniteElementState> fields;
   /// Map of dual vectors used by the solver
   std::map<std::string, FiniteElementDual> duals;

   /// Reference to solver state vector
   FiniteElementState &state() { return fields.at("state"); }
   const FiniteElementState &state() const { return fields.at("state"); }
   /// Reference to solver adjoint vector
   FiniteElementState &adjoint() { return fields.at("adjoint"); }
   const FiniteElementState &adjoint() const { return fields.at("adjoint"); }
   /// Reference to solver residual dual vec
   FiniteElementDual &res_vec() { return duals.at("residual"); }
   const FiniteElementDual &res_vec() const { return duals.at("residual"); }

   /// Reference to the state vectors finite element space
   mfem::ParFiniteElementSpace &fes() { return state().space(); }
   const mfem::ParFiniteElementSpace &fes() const { return state().space(); }

   void setUpExternalFields();

   void setState_(std::any function,
                  const std::string &name,
                  mfem::Vector &state) override;

   double calcStateError_(std::any ex_sol,
                          const std::string &name,
                          const mfem::Vector &state) override;

   /// For code that should be executed before the time stepping begins
   /// \param[in] state - the current state
   /// \note This is `final` because we want to ensure the `state` Vector gets
   /// associated with the state field.  This association may not happen if the
   /// client overwrites this definition; however, there is a call to the
   /// virtual function derivedPDEinitialHook(state) that the client can
   /// overwrite.
   virtual void initialHook(const mfem::Vector &state) override final;

   /// Code in a derived class that should be executed before time-stepping
   /// \param[in] state - the current state
   virtual void derivedPDEInitialHook(const mfem::Vector &state) { }

   /// For code that should be executed before `ode_solver->Step`
   /// \param[in] iter - the current iteration
   /// \param[in] t - the current time (before the step)
   /// \param[in] dt - the step size that will be taken
   /// \param[in] state - the current state
   /// \note This is `final` because we want to ensure that
   /// AbstractSolver2::iterationHook() is called.
   virtual void iterationHook(int iter,
                              double t,
                              double dt,
                              const mfem::Vector &state) override final;

   /// Code in a derived class that should be executed each time step
   /// \param[in] iter - the current iteration
   /// \param[in] t - the current time (before the step)
   /// \param[in] dt - the step size that will be taken
   /// \param[in] state - the current state
   virtual void derivedPDEIterationHook(int iter,
                                        double t,
                                        double dt,
                                        const mfem::Vector &state)
   { }

   /// For code that should be executed after the time stepping ends
   /// \param[in] iter - the terminal iteration
   /// \param[in] t_final - the final time
   /// \param[in] state - the current state
   virtual void terminalHook(int iter,
                             double t_final,
                             const mfem::Vector &state) override final;

   /// Code in a derived class that should be executed after time stepping ends
   /// \param[in] iter - the terminal iteration
   /// \param[in] t_final - the final time
   /// \param[in] state - the current state
   virtual void derivedPDETerminalHook(int iter,
                                       double t_final,
                                       const mfem::Vector &state)
   { }
};

}  // namespace miso

#endif
