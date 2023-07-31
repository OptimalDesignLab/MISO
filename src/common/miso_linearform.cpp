#include <vector>

#include "mfem.hpp"

#include "miso_input.hpp"
#include "miso_integrator.hpp"
#include "miso_linearform.hpp"

namespace miso
{
void setInputs(MISOLinearForm &load, const MISOInputs &inputs)
{
   for (const auto &in : inputs)
   {
      const auto &input = in.second;
      if (input.isField())
      {
         const auto &name = in.first;
         auto it = load.lf_fields->find(name);
         if (it != load.lf_fields->end())
         {
            auto &field = it->second;
            field.GetTrueVector().SetDataAndSize(
                input.getField(), field.ParFESpace()->GetTrueVSize());
            field.SetFromTrueVector();
         }
      }
   }
   setInputs(load.integs, inputs);
}

void setOptions(MISOLinearForm &load, const nlohmann::json &options)
{
   setOptions(load.integs, options);
}

void addLoad(MISOLinearForm &load, mfem::Vector &tv)
{
   load.lf.Assemble();
   load.lf.ParallelAssemble(load.scratch);
   add(tv, load.scratch, tv);
}

double vectorJacobianProduct(MISOLinearForm &load,
                             const mfem::HypreParVector &load_bar,
                             const std::string &wrt)
{
   if (load.scalar_sens.count(wrt) != 0)
   {
      throw std::logic_error(
          "vectorJacobianProduct not implemented for MISOLinearForm!\n");
      // auto &adjoint = load.lf_fields->at("adjoint");
      // adjoint = load_bar;
      // return load.scalar_sens.at(wrt).GetEnergy();
   }
   else
   {
      return 0.0;
   }
}

void vectorJacobianProduct(MISOLinearForm &load,
                           const mfem::HypreParVector &load_bar,
                           const std::string &wrt,
                           mfem::HypreParVector &wrt_bar)
{
   if (load.sens.count(wrt) != 0)
   {
      auto &adjoint = load.lf_fields->at("adjoint");
      adjoint = load_bar;
      load.sens.at(wrt).Assemble();
      load.sens.at(wrt).ParallelAssemble(load.scratch);
      wrt_bar += load.scratch;
   }
}

}  // namespace miso
