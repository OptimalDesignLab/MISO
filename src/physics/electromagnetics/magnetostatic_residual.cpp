#include <string>

#include "mfem.hpp"
#include "nlohmann/json.hpp"

#include "mach_input.hpp"

#include "magnetostatic_residual.hpp"

namespace mach
{
int getSize(const MagnetostaticResidual &residual)
{
   return getSize(residual.res);
}

void setInputs(MagnetostaticResidual &residual, const mach::MachInputs &inputs)
{
   setInputs(residual.res, inputs);
   setInputs(residual.load, inputs);
}

void setOptions(MagnetostaticResidual &residual, const nlohmann::json &options)
{
   setOptions(residual.res, options);
   setOptions(residual.load, options);
}

void evaluate(MagnetostaticResidual &residual,
              const mach::MachInputs &inputs,
              mfem::Vector &res_vec)
{
   evaluate(residual.res, inputs, res_vec);
   setInputs(residual.load, inputs);
   addLoad(residual.load, res_vec);

   // mfem::Vector state;
   // setVectorFromInputs(inputs, "state", state);
   // const auto &ess_tdofs = residual.res.getEssentialDofs();
   // for (int i = 0; i < ess_tdofs.Size(); ++i)
   // {
   //    res_vec(ess_tdofs[i]) = state(ess_tdofs[i]);
   // }
}

void linearize(MagnetostaticResidual &residual, const mach::MachInputs &inputs)
{
   linearize(residual.res, inputs);
}

mfem::Operator &getJacobian(MagnetostaticResidual &residual,
                            const mach::MachInputs &inputs,
                            const std::string &wrt)
{
   return getJacobian(residual.res, inputs, wrt);
}

mfem::Operator &getJacobianTranspose(MagnetostaticResidual &residual,
                                     const mach::MachInputs &inputs,
                                     const std::string &wrt)
{
   return getJacobianTranspose(residual.res, inputs, wrt);
}

void setUpAdjointSystem(MagnetostaticResidual &residual,
                        mfem::Solver &adj_solver,
                        const mach::MachInputs &inputs,
                        mfem::Vector &state_bar,
                        mfem::Vector &adjoint)
{
   setUpAdjointSystem(residual.res, adj_solver, inputs, state_bar, adjoint);
}

double jacobianVectorProduct(MagnetostaticResidual &residual,
                             const mfem::Vector &wrt_dot,
                             const std::string &wrt)
{
   auto res_dot = jacobianVectorProduct(residual.res, wrt_dot, wrt);
   res_dot += jacobianVectorProduct(residual.load, wrt_dot, wrt);
   return res_dot;
}

void jacobianVectorProduct(MagnetostaticResidual &residual,
                           const mfem::Vector &wrt_dot,
                           const std::string &wrt,
                           mfem::Vector &res_dot)
{
   jacobianVectorProduct(residual.res, wrt_dot, wrt, res_dot);
   jacobianVectorProduct(residual.load, wrt_dot, wrt, res_dot);
}

double vectorJacobianProduct(MagnetostaticResidual &residual,
                             const mfem::Vector &res_bar,
                             const std::string &wrt)
{
   auto wrt_bar = vectorJacobianProduct(residual.res, res_bar, wrt);
   wrt_bar += vectorJacobianProduct(residual.load, res_bar, wrt);
   return wrt_bar;
}

void vectorJacobianProduct(MagnetostaticResidual &residual,
                           const mfem::Vector &res_bar,
                           const std::string &wrt,
                           mfem::Vector &wrt_bar)
{
   vectorJacobianProduct(residual.res, res_bar, wrt, wrt_bar);
   vectorJacobianProduct(residual.load, res_bar, wrt, wrt_bar);
}

mfem::Solver *getPreconditioner(MagnetostaticResidual &residual)
{
   return residual.prec.get();
}

}  // namespace mach
