#include "magnetostatic_residual.hpp"

    int getSize(const MagnetostaticResidual &residual)
   {
      return getSize(residual.res);
   }

    void setInputs(MagnetostaticResidual &residual, const mach::MachInputs &inputs)
   {
      setInputs(residual.res, inputs);
      setInputs(*residual.load, inputs);
   }

    void setOptions(MagnetostaticResidual &residual, const nlohmann::json &options)
   {
      setOptions(residual.res, options);
      setOptions(*residual.load, options);
   }

    void evaluate(MagnetostaticResidual &residual,
                        const mach::MachInputs &inputs,
                        mfem::Vector &res_vec)
   {
      evaluate(residual.res, inputs, res_vec);
      setInputs(*residual.load, inputs);
      addLoad(*residual.load, res_vec);
   }

    mfem::Operator &getJacobian(MagnetostaticResidual &residual,
                                      const mach::MachInputs &inputs,
                                      std::string wrt)
   {
      return getJacobian(residual.res, inputs, std::move(wrt));
   }

    mfem::Solver *getPreconditioner(MagnetostaticResidual &residual)
   {
      return residual.prec.get();
   }