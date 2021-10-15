#include <vector>

#include "mfem.hpp"

#include "mach_input.hpp"
#include "mach_integrator.hpp"
#include "mach_linearform.hpp"

namespace mach
{
void setInputs(MachLinearForm &load, const MachInputs &inputs)
{
   setInputs(load.integs, inputs);
}

void addLoad(MachLinearForm &load, mfem::Vector &tv)
{
   load.lf.Assemble();
   load.lf.ParallelAssemble(load.scratch);
   add(tv, load.scratch, tv);
}

double vectorJacobianProduct(MachLinearForm &load,
                             const mfem::HypreParVector &load_bar,
                             const std::string &wrt)
{
   if (load.scalar_sens.count(wrt) != 0)
   {
      throw std::logic_error(
          "vectorJacobianProduct not implemented for MachLinearForm!\n");
      // auto &adjoint = load.lf_fields.at("adjoint");
      // adjoint = load_bar;
      // return load.scalar_sens.at(wrt).GetEnergy();
   }
   else
   {
      return 0.0;
   }
}

void vectorJacobianProduct(MachLinearForm &load,
                           const mfem::HypreParVector &load_bar,
                           const std::string &wrt,
                           mfem::HypreParVector &wrt_bar)
{
   if (load.sens.count(wrt) != 0)
   {
      auto &adjoint = load.lf_fields.at("adjoint");
      adjoint = load_bar;
      load.sens.at(wrt).Assemble();
      load.sens.at(wrt).ParallelAssemble(load.scratch);
      wrt_bar += load.scratch;
   }
}

}  // namespace mach
