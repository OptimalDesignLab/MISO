#ifndef MISO_MAGNETOSTATIC_LOAD
#define MISO_MAGNETOSTATIC_LOAD

#include <map>
#include <memory>
#include <optional>
#include <string>

#include "mfem.hpp"
#include "nlohmann/json.hpp"

#include "current_load.hpp"
#include "finite_element_state.hpp"
#include "miso_input.hpp"
#include "magnetic_load.hpp"

namespace miso
{
class MagnetostaticLoad final
{
public:
   friend inline void setInputs(MagnetostaticLoad &load,
                                const MISOInputs &inputs)
   {
      if (load.current_load.has_value())
      {
         setInputs(*load.current_load.value(), inputs);
      }
      if (load.magnetic_load.has_value())
      {
         setInputs(load.magnetic_load.value(), inputs);
      }
   }

   friend inline void setOptions(MagnetostaticLoad &load,
                                 const nlohmann::json &options)
   {
      if (load.current_load.has_value())
      {
         setOptions(*load.current_load.value(), options);
      }
      if (load.magnetic_load.has_value())
      {
         setOptions(load.magnetic_load.value(), options);
      }
   }

   friend inline void addLoad(MagnetostaticLoad &load, mfem::Vector &tv)
   {
      if (load.current_load.has_value())
      {
         addLoad(*load.current_load.value(), tv);
      }
      if (load.magnetic_load.has_value())
      {
         addLoad(load.magnetic_load.value(), tv);
      }
   }

   friend inline double vectorJacobianProduct(MagnetostaticLoad &load,
                                              const mfem::Vector &res_bar,
                                              const std::string &wrt)
   {
      double wrt_bar = 0.0;
      if (load.current_load.has_value())
      {
         wrt_bar +=
             vectorJacobianProduct(*load.current_load.value(), res_bar, wrt);
      }
      if (load.magnetic_load.has_value())
      {
         wrt_bar +=
             vectorJacobianProduct(load.magnetic_load.value(), res_bar, wrt);
      }
      return wrt_bar;
   }

   friend inline void vectorJacobianProduct(MagnetostaticLoad &load,
                                            const mfem::Vector &res_bar,
                                            const std::string &wrt,
                                            mfem::Vector &wrt_bar)
   {
      if (load.current_load.has_value())
      {
         vectorJacobianProduct(
             *load.current_load.value(), res_bar, wrt, wrt_bar);
      }
      if (load.magnetic_load.has_value())
      {
         vectorJacobianProduct(
             load.magnetic_load.value(), res_bar, wrt, wrt_bar);
      }
   }

   MagnetostaticLoad(adept::Stack &diff_stack,
                     mfem::ParFiniteElementSpace &fes,
                     std::map<std::string, FiniteElementState> &fields,
                     const nlohmann::json &options,
                     const nlohmann::json &materials,
                     mfem::Coefficient &nu)
    : current_load(
          [&]() -> std::optional<std::unique_ptr<CurrentLoad>>
          {
             if (options.contains("current"))
             {
                return std::make_unique<CurrentLoad>(
                    diff_stack, fes, fields, options);
             }
             else
             {
                return std::nullopt;
             }
          }()),
      magnetic_load(
          [&]() -> std::optional<MagneticLoad>
          {
             if (options.contains("magnets"))
             {
                return MagneticLoad(
                    diff_stack, fes, fields, options, materials, nu);
             }
             else
             {
                return std::nullopt;
             }
          }())
   { }

private:
   std::optional<std::unique_ptr<CurrentLoad>> current_load;
   std::optional<MagneticLoad> magnetic_load;
};

}  // namespace miso

#endif
