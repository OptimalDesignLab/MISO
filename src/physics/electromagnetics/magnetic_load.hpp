#ifndef MACH_MAGNETIC_LOAD
#define MACH_MAGNETIC_LOAD

#include "mfem.hpp"

#include "pfem_extras.hpp"
#include "coefficient.hpp"
#include "mach_input.hpp"

namespace mach
{

class MagneticLoad final
{
public:
   /// Used to set scalar inputs in the underlying load type
   /// Ends up calling `setInputs` on either the `MachLinearForm` or
   /// a specialized version for each particular load.
   friend void setInputs(MagneticLoad &load,
                         const MachInputs &inputs);

   /// Assemble the load vector on the true dofs and store in tv
   friend void addLoad(MagneticLoad &load,
                       mfem::Vector &tv);

   friend double vectorJacobianProduct(MagneticLoad &load,
                                       const mfem::HypreParVector &res_bar,
                                       std::string wrt);

   friend void vectorJacobianProduct(MagneticLoad &load,
                                     const mfem::HypreParVector &res_bar,
                                     std::string wrt,
                                     mfem::HypreParVector &wrt_bar);

   MagneticLoad(mfem::ParFiniteElementSpace &pfes,
                mfem::VectorCoefficient &mag_coeff,
                mfem::Coefficient &nu);

private:
   mfem::ParFiniteElementSpace &fes;
   mfem::RT_FECollection rt_coll;
   mfem::ParFiniteElementSpace rt_fes;
   
   mfem::VectorCoefficient &mag_coeff;
   mfem::HypreParVector load;

   mfem::ParMixedBilinearForm weakCurlMuInv;
   mfem::ParGridFunction M;
   mfem::ParGridFunction scratch;

   /// flag to know if the load vector should be reassembled
   bool dirty;

   /// Assemble the magnetic source load vector
   void assembleLoad();
};

} // namespace mach

#endif
