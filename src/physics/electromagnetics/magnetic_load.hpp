#ifndef MACH_MAGNETIC_LOAD
#define MACH_MAGNETIC_LOAD

#include "mfem.hpp"

#include "coefficient.hpp"
#include "mach_input.hpp"
#include "mach_linearform.hpp"

namespace mach
{
class MagneticLoad final
{
public:
   friend inline void setInputs(MagneticLoad &load, const MachInputs &inputs)
   {
      setInputs(load.lf, inputs);
   }

   /// Assemble the load vector on the true dofs and store in tv
   friend inline void addLoad(MagneticLoad &load, mfem::Vector &tv)
   {
      addLoad(load.lf, tv);
   }

   friend inline double vectorJacobianProduct(
       MagneticLoad &load,
       const mfem::HypreParVector &load_bar,
       const std::string &wrt)
   {
      return vectorJacobianProduct(load.lf, load_bar, wrt);
   }

   friend inline void vectorJacobianProduct(
       MagneticLoad &load,
       const mfem::HypreParVector &load_bar,
       const std::string &wrt,
       mfem::HypreParVector &wrt_bar)
   {
      vectorJacobianProduct(load.lf, load_bar, wrt, wrt_bar);
   }

   MagneticLoad(mfem::ParFiniteElementSpace &pfes,
                mfem::VectorCoefficient &mag_coeff,
                mfem::Coefficient &nu);

private:
   std::unordered_map<std::string, mfem::ParGridFunction> mag_load_fields;
   MachLinearForm lf;
   mfem::ScalarVectorProductCoefficient nuM;
};

class LegacyMagneticLoad final
{
public:
   /// Used to set scalar inputs in the underlying load type
   /// Ends up calling `setInputs` on either the `MachLinearForm` or
   /// a specialized version for each particular load.
   friend void setInputs(LegacyMagneticLoad &load, const MachInputs &inputs);

   /// Assemble the load vector on the true dofs and store in tv
   friend void addLoad(LegacyMagneticLoad &load, mfem::Vector &tv);

   friend double vectorJacobianProduct(LegacyMagneticLoad &load,
                                       const mfem::HypreParVector &res_bar,
                                       const std::string &wrt);

   friend void vectorJacobianProduct(LegacyMagneticLoad &load,
                                     const mfem::HypreParVector &res_bar,
                                     const std::string &wrt,
                                     mfem::HypreParVector &wrt_bar);

   LegacyMagneticLoad(mfem::ParFiniteElementSpace &pfes,
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

}  // namespace mach

#endif
