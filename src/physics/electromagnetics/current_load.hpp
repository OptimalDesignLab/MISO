#ifndef MISO_CURRENT_LOAD
#define MISO_CURRENT_LOAD

#include "mfem.hpp"
#include "nlohmann/json.hpp"

#include "current_source_functions.hpp"
#include "div_free_projector.hpp"
#include "finite_element_state.hpp"
#include "miso_input.hpp"

namespace miso
{
class VectorFEMassIntegratorMeshSens;
class VectorFEDomainLFIntegratorMeshSens;

class CurrentLoad final
{
public:
   friend int getSize(const CurrentLoad &load);

   /// Used to set scalar inputs in the underlying load type
   /// Ends up calling `setInputs` on either the `MISOLinearForm` or
   /// a specialized version for each particular load.
   friend void setInputs(CurrentLoad &load, const MISOInputs &inputs);

   friend void setOptions(CurrentLoad &load, const nlohmann::json &options);

   /// Assemble the load vector on the true dofs and store in tv
   friend void addLoad(CurrentLoad &load, mfem::Vector &tv);

   friend double vectorJacobianProduct(CurrentLoad &load,
                                       const mfem::Vector &load_bar,
                                       const std::string &wrt);

   friend void vectorJacobianProduct(CurrentLoad &load,
                                     const mfem::Vector &load_bar,
                                     const std::string &wrt,
                                     mfem::Vector &wrt_bar);

   CurrentLoad(adept::Stack &diff_stack,
               mfem::ParFiniteElementSpace &fes,
               std::map<std::string, FiniteElementState> &fields,
               const nlohmann::json &options);

private:
   /// Coefficient to represent current density
   CurrentDensityCoefficient current;
   /// Finite element spaces and collections needed for divergence cleaning
   mfem::ParFiniteElementSpace &fes;
   mfem::H1_FECollection h1_coll;
   mfem::ParFiniteElementSpace h1_fes;
   mfem::RT_FECollection rt_coll;
   mfem::ParFiniteElementSpace rt_fes;

   mfem::ParBilinearForm nd_mass;
   mfem::ParLinearForm J;
   mfem::ParGridFunction j;
   mfem::ParGridFunction div_free_current_vec;
   mfem::ParGridFunction scratch;
   // mfem::Vector scratch;
   mfem::Vector load;

   mfem::Vector X;
   mfem::Vector RHS;
   mfem::Array<int> dummmy_ess_tdof_list;
   mfem::HypreBoomerAMG amg;
   std::unique_ptr<mfem::Solver> pcg;

   DivergenceFreeProjector div_free_proj;

   mfem::ParLinearForm mesh_sens;
   VectorFEMassIntegratorMeshSens *m_j_mesh_sens;
   VectorFEDomainLFIntegratorMeshSens *J_mesh_sens;
   VectorFEMassIntegratorMeshSens *m_l_mesh_sens;

   /// essential tdofs
   mfem::Array<int> ess_tdof_list;

   /// flag to know if the load vector should be reassembled
   bool dirty;

   /// Assemble the divergence free load vector
   void assembleLoad();
};

}  // namespace miso

#endif
