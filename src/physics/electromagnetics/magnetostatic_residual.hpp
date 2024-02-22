#ifndef MISO_MAGNETOSTATIC_RESIDUAL
#define MISO_MAGNETOSTATIC_RESIDUAL

#include <map>
#include <memory>
#include <optional>
#include <string>

#include "adept.h"
#include "miso_load.hpp"
#include "mfem.hpp"
#include "nlohmann/json.hpp"

#include "electromag_integ.hpp"
#include "miso_input.hpp"
#include "miso_residual.hpp"
#include "miso_nonlinearform.hpp"
#include "magnetostatic_load.hpp"

namespace miso
{
class MagnetostaticResidual final
{
public:
   friend int getSize(const MagnetostaticResidual &residual);

   friend void setInputs(MagnetostaticResidual &residual,
                         const MISOInputs &inputs);

   friend void setOptions(MagnetostaticResidual &residual,
                          const nlohmann::json &options);

   friend void evaluate(MagnetostaticResidual &residual,
                        const MISOInputs &inputs,
                        mfem::Vector &res_vec);

   friend void linearize(MagnetostaticResidual &residual,
                         const miso::MISOInputs &inputs);

   friend mfem::Operator &getJacobian(MagnetostaticResidual &residual,
                                      const MISOInputs &inputs,
                                      const std::string &wrt);

   friend mfem::Operator &getJacobianTranspose(MagnetostaticResidual &residual,
                                               const miso::MISOInputs &inputs,
                                               const std::string &wrt);

   friend void setUpAdjointSystem(MagnetostaticResidual &residual,
                                  mfem::Solver &adj_solver,
                                  const miso::MISOInputs &inputs,
                                  mfem::Vector &state_bar,
                                  mfem::Vector &adjoint);

   friend void finalizeAdjointSystem(MagnetostaticResidual &residual,
                                     mfem::Solver &adj_solver,
                                     const miso::MISOInputs &inputs,
                                     mfem::Vector &state_bar,
                                     mfem::Vector &adjoint);

   friend double jacobianVectorProduct(MagnetostaticResidual &residual,
                                       const mfem::Vector &wrt_dot,
                                       const std::string &wrt);

   friend void jacobianVectorProduct(MagnetostaticResidual &residual,
                                     const mfem::Vector &wrt_dot,
                                     const std::string &wrt,
                                     mfem::Vector &res_dot);

   friend double vectorJacobianProduct(MagnetostaticResidual &residual,
                                       const mfem::Vector &res_bar,
                                       const std::string &wrt);

   friend void vectorJacobianProduct(MagnetostaticResidual &residual,
                                     const mfem::Vector &res_bar,
                                     const std::string &wrt,
                                     mfem::Vector &wrt_bar);

   friend mfem::Solver *getPreconditioner(MagnetostaticResidual &residual);

   MagnetostaticResidual(adept::Stack &diff_stack,
                         mfem::ParFiniteElementSpace &fes,
                         std::map<std::string, FiniteElementState> &fields,
                         const nlohmann::json &options,
                         const nlohmann::json &materials,
                         StateCoefficient &nu);

private:
   /// Nonlinear form that handles the curl curl term of the weak form
   MISONonlinearForm res;
   /// coefficient for weakly imposed boundary conditions
   std::unique_ptr<mfem::Coefficient> g;
   /// Load vector for current and magnetic sources
   // MagnetostaticLoad load;
   std::unique_ptr<MISOLoad> load;
   std::unique_ptr<CurrentDensityCoefficient2D> current_coeff;
   std::unique_ptr<MagnetizationCoefficient> mag_coeff;
   // std::unique_ptr<mfem::ScalarVectorProductCoefficient> nuM; // No. Needs to
   // be a miso::ScalarVectorProductCoefficient rather than an
   // mfem::ScalarVectorProductCoefficient now
   std::unique_ptr<miso::ScalarVectorProductCoefficient>
       nuM;  // Yes. Needs to be a miso::ScalarVectorProductCoefficient rather
             // than an mfem::ScalarVectorProductCoefficient now

   /// preconditioner for inverting residual's state Jacobian
   std::unique_ptr<mfem::Solver> prec;

   /// Work vector
   mfem::Vector scratch;
};

}  // namespace miso

#endif
