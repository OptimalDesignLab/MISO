#ifndef MACH_MAGNETOSTATIC_RESIDUAL
#define MACH_MAGNETOSTATIC_RESIDUAL

#include <memory>
#include <mpi.h>

#include "mfem.hpp"
#include "nlohmann/json.hpp"

#include "electromag_integ.hpp"
#include "mach_input.hpp"
#include "mach_nonlinearform.hpp"
#include "magnetostatic_load.hpp"

// namespace mach
// {
// class MagnetostaticResidual final
// {
// public:
//    friend int getSize(const MagnetostaticResidual &residual);

//    friend void setInputs(MagnetostaticResidual &residual,
//                          const MachInputs &inputs);

//    friend void setOptions(MagnetostaticResidual &residual,
//                           const nlohmann::json &options);

//    friend void evaluate(MagnetostaticResidual &residual,
//                         const MachInputs &inputs,
//                         mfem::Vector &res_vec);

//    friend mfem::Operator &getJacobian(MagnetostaticResidual &residual,
//                                       const MachInputs &inputs,
//                                       const std::string &wrt);

//    MagnetostaticResidual(mfem::ParFiniteElementSpace &pfes,
//                          mfem::VectorCoefficient &mag_coeff,
//                          StateCoefficient &nu)
//     : nlf(pfes, fields), load(pfes, current_coeff, mag_coeff, nu)
//    {
//       nlf.addDomainIntegrator(new CurlCurlNLFIntegrator(nu));
//    }

// private:
//    MachNonlinearForm nlf;
//    MagnetostaticLoad load;
// };

}  // namespace mach

#endif
