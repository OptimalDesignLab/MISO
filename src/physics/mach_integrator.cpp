#include <string>

#include "mfem.hpp"

#include "mach_input.hpp"
#include "mach_integrator.hpp"

namespace mach
{

void setScalarInputs(std::vector<MachIntegrator> &integrators,
                    const MachInputs &inputs)
{
   for (auto &input : inputs)
   {
      setScalarInput(integrators, input.first, input.second);
   }
}

void setScalarInput(std::vector<MachIntegrator> &integrators,
                    const std::string &name,
                    const MachInput &input)
{
   for (auto &integrator : integrators)
   {
      setInput(integrator, name, input);
   }
}

void setInput(MachIntegrator &integ,
              const std::string &name,
              const MachInput &input)
{
   integ.self_->setInput_(name, input);
}

void setInput(mfem::NonlinearFormIntegrator &integ,
              const std::string &name,
              const MachInput &input)
{
   // do nothing for default integrator
}

void setInput(mfem::LinearFormIntegrator &integ,
              const std::string &name,
              const MachInput &input)
{
   // do nothing for default integrator
}

//// notes: for some reason MachIntegrator is not specializing 
/// for a LinearFormIntegrator... test for mach_load is failing since
/// the integrator isn't being updated

/**********************************************************
/// the issue is that I'm taking the pointers to linearform objects and 
/// constructing mach integs from them. I need to create the MachIntegs when
/// I add the integrators while they still have their type, otherwise the
/// machinteg doesn't know to call the correct template specialization
**********************************************************/

/// might need to use the solver to keep around the vector of mach integs?
/// subclass MachLoad for LinearFormLoad to keep the vector in there?
/// then I need to use pointers to loads :(

} // namespace mach
