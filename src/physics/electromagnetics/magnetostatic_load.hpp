#ifndef MACH_MAGNETOSTATIC_LOAD
#define MACH_MAGNETOSTATIC_LOAD

#include <memory>
#include <mpi.h>

#include "mfem.hpp"
#include "nlohmann/json.hpp"

#include "current_load.hpp"
#include "mach_input.hpp"
#include "magnetic_load.hpp"

namespace mach
{
class MagnetostaticLoad final
{
public:
   friend void setInputs(MagnetostaticLoad &load, const MachInputs &inputs);

   friend void setOptions(MagnetostaticLoad &load,
                          const nlohmann::json &options);

   friend void addLoad(MagnetostaticLoad &load, mfem::Vector &tv);

   friend double vectorJacobianProduct(MagnetostaticLoad &load,
                                       const mfem::Vector &res_bar,
                                       const std::string &wrt);

   friend void vectorJacobianProduct(MagnetostaticLoad &load,
                                     const mfem::Vector &res_bar,
                                     const std::string &wrt,
                                     mfem::Vector &wrt_bar);

   MagnetostaticLoad(mfem::ParFiniteElementSpace &pfes, mfem::Coefficient &nu)
    : current_load(pfes, options, current_coeff),
      magnetic_load(pfes, mag_coeff, nu)
   { }

private:
   nlohmann::json options;
   CurrentLoad current_load;
   MagneticLoad magnetic_load;
};

}  // namespace mach

#endif
