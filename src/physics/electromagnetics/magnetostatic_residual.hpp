#ifndef MACH_MAGNETOSTATIC_RESIDUAL
#define MACH_MAGNETOSTATIC_RESIDUAL

#include <map>
#include <memory>
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

   friend mfem::Solver *getPreconditioner(MagnetostaticResidual &residual);

   MagnetostaticResidual(adept::Stack &diff_stack,
                         mfem::ParFiniteElementSpace &fes,
                         std::map<std::string, FiniteElementState> &fields,
                         const nlohmann::json &options,
                         const nlohmann::json &materials)
    : nu(options, materials),
      nlf(fes, old_fields),
      load(diff_stack, fes, fields, options, materials, nu),
      prec(constructPreconditioner(fes, options["lin-prec"]))

   {
      nlf.addDomainIntegrator(new CurlCurlNLFIntegrator(nu));
   }

private:
   /// Coefficient representing the potentially nonlinear magnetic reluctivity
   ReluctivityCoefficient nu;
   std::unordered_map<std::string, mfem::ParGridFunction> old_fields;
   /// Nonlinear form that handles the curl curl term of the weak form
   MachNonlinearForm nlf;
   /// Load vector for current and magnetic sources
   MagnetostaticLoad load;

   /// preconditioner for inverting residual's state Jacobian
   std::unique_ptr<mfem::Solver> prec;

   std::unique_ptr<mfem::Solver> constructPreconditioner(
       mfem::ParFiniteElementSpace &fes,
       const nlohmann::json &prec_options)
   {
      auto ams = std::make_unique<mfem::HypreAMS>(&fes);
      ams->SetPrintLevel(prec_options["printlevel"].get<int>());
      ams->SetSingularProblem();
      return ams;
   }
};

}  // namespace mach

#endif
