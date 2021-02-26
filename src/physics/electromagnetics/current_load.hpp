#ifndef MACH_CURRENT_LOAD
#define MACH_CURRENT_LOAD

#include "mfem.hpp"

#include "pfem_extras.hpp"
#include "coefficient.hpp"
#include "mach_input.hpp"

namespace mach
{

class CurrentLoad
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

   CurrentLoad(mfem::ParFiniteElementSpace &pfes,
               mfem::VectorCoefficient &current_coeff);

private:
   mfem::ParFiniteElementSpace &fes;
   mfem::H1_FECollection h1_coll;
   mfem::ParFiniteElementSpace h1_fes;
   mfem::RT_FECollection rt_coll;
   mfem::ParFiniteElementSpace rt_fes;
   
   double current_density;
   mfem::HypreParVector load;
   mfem::HypreParVector scratch;

   mfem::ParBilinearForm nd_mass;
   mfem::ParLinearForm J;
   mfem::ParGridFunction j;
   mfem::ParGridFunction div_free_current_vec;
   mfem::common::DivergenceFreeProjector div_free_proj;

   /// Assemble the divergence free load vector
   void assembleLoad();
};

} // namespace mach

#endif
