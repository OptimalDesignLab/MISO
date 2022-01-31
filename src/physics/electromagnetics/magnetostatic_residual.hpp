#ifndef MACH_MAGNETOSTATIC_RESIDUAL
#define MACH_MAGNETOSTATIC_RESIDUAL

#include <map>
#include <string>

#include "mfem.hpp"
#include "nlohmann/json.hpp"

#include "electromag_integ.hpp"
#include "mach_input.hpp"
#include "mach_nonlinearform.hpp"
#include "magnetostatic_load.hpp"
#include "reluctivity_coefficient.hpp"

namespace mach
{
class MagnetostaticResidual final
{
public:
   friend int getSize(const MagnetostaticResidual &residual);

   friend void setInputs(MagnetostaticResidual &residual,
                         const MachInputs &inputs);

   friend void setOptions(MagnetostaticResidual &residual,
                          const nlohmann::json &options);

   friend void evaluate(MagnetostaticResidual &residual,
                        const MachInputs &inputs,
                        mfem::Vector &res_vec);

   friend mfem::Operator &getJacobian(MagnetostaticResidual &residual,
                                      const MachInputs &inputs,
                                      const std::string &wrt);

   MagnetostaticResidual(adept::Stack &diff_stack,
                         mfem::ParFiniteElementSpace &fespace,
                         std::map<std::string, FintieElementState> &fields,
                         const nlohmann::json &options,
                         const nlohmann::json &materials)
    : nu(options, materials),
      nlf(fespace, old_fields),
      load(diff_stack, fespace, fields, options, materials, nu)
   {
      nlf.addDomainIntegrator(new CurlCurlNLFIntegrator(nu));
   }

private:
   ReluctivityCoefficient nu;
   std::unordered_map<std::string, mfem::ParGridFunction> old_fields;
   MachNonlinearForm nlf;
   MagnetostaticLoad load;
};

}  // namespace mach

#endif
