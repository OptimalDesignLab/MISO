#ifndef MACH_MAGNETOSTATIC_LOAD
#define MACH_MAGNETOSTATIC_LOAD

#include <map>
#include <memory>
#include <string>

#include "mfem.hpp"
#include "nlohmann/json.hpp"

#include "current_load.hpp"
#include "finite_element_state.hpp"
#include "mach_input.hpp"
#include "magnetic_load.hpp"

namespace mach
{
class MagnetostaticLoad final
{
public:
   friend inline void setInputs(MagnetostaticLoad &load,
                                const MachInputs &inputs)
   {
      setInputs(load.current_load, inputs);
      setInputs(load.magnetic_load, inputs);
   }

   friend inline void setOptions(MagnetostaticLoad &load,
                                 const nlohmann::json &options)
   {
      setOptions(load.current_load, options);
      setOptions(load.magnetic_load, options);
   }

   friend inline void addLoad(MagnetostaticLoad &load, mfem::Vector &tv)
   {
      addLoad(load.current_load, tv);
      addLoad(load.magnetic_load, tv);
   }

   friend inline double vectorJacobianProduct(MagnetostaticLoad &load,
                                              const mfem::Vector &res_bar,
                                              const std::string &wrt)
   {
      double wrt_bar = 0.0;
      wrt_bar += vectorJacobianProduct(load.current_load, res_bar, wrt);
      wrt_bar += vectorJacobianProduct(load.magnetic_load, res_bar, wrt);
      return wrt_bar;
   }

   friend inline void vectorJacobianProduct(MagnetostaticLoad &load,
                                            const mfem::Vector &res_bar,
                                            const std::string &wrt,
                                            mfem::Vector &wrt_bar)
   {
      vectorJacobianProduct(load.current_load, res_bar, wrt, wrt_bar);
      vectorJacobianProduct(load.magnetic_load, res_bar, wrt, wrt_bar);
   }

   MagnetostaticLoad(adept::Stack &diff_stack,
                     mfem::ParFiniteElementSpace &fes,
                     std::map<std::string, FiniteElementState> &fields,
                     const nlohmann::json &options,
                     const nlohmann::json &materials,
                     mfem::Coefficient &nu)
    : current_load(diff_stack, fes, fields, options),
      magnetic_load(diff_stack, fes, fields, options, materials, nu)
   { }

private:
   CurrentLoad current_load;
   MagneticLoad magnetic_load;
};

}  // namespace mach

#endif
