#ifndef MACH_CURRENT_LOAD
#define MACH_CURRENT_LOAD

#include "mfem.hpp"

#include "div_free_projector.hpp"
#include "pfem_extras.hpp"
#include "mach_input.hpp"

namespace mach
{

class CurrentLoad final
{
public:
   /// Used to set scalar inputs in the underlying load type
   /// Ends up calling `setInputs` on either the `MachLinearForm` or
   /// a specialized version for each particular load.
   friend void setInputs(CurrentLoad &load,
                         const MachInputs &inputs);

   /// Assemble the load vector on the true dofs and store in tv
   friend void addLoad(CurrentLoad &load,
                       mfem::Vector &tv);

   friend double vectorJacobianProduct(CurrentLoad &load,
                                       const mfem::HypreParVector &res_bar,
                                       std::string wrt);

   friend void vectorJacobianProduct(CurrentLoad &load,
                                     const mfem::HypreParVector &res_bar,
                                     std::string wrt,
                                     mfem::HypreParVector &wrt_bar);

   CurrentLoad(mfem::ParFiniteElementSpace &pfes,
               mfem::VectorCoefficient &current_coeff);

private:
   double current_density;
   /// Coefficient to represent current_density*current_coeff
   mfem::ScalarVectorProductCoefficient current;
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
   // mfem::ParGridFunction scratch;
   mfem::HypreParVector scratch;
   mfem::HypreParVector load;

   DivergenceFreeProjector div_free_proj;
   // mfem::common::DivergenceFreeProjector div_free_proj;

   /// flag to know if the load vector should be reassembled
   bool dirty;

   /// Assemble the divergence free load vector
   void assembleLoad();
};

} // namespace mach

#endif
