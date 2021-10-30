#include "utils.hpp"
#include "mach_input.hpp"

namespace mach
{

void setValueFromInput(const MachInput &input, double &value)
{
   value = std::get<double>(input);
}

void setValueFromInputs(const MachInputs &inputs,
                        const std::string &key,
                        double &value,
                        bool error_if_not_found)
{
   auto input = inputs.find(key);
   if (input != inputs.end())
   {
      setValueFromInput(input->second, value);
   }
   else if (error_if_not_found)
   {
      throw MachException("key = " + key + "not found in inputs!\n");
   }
}

void setVectorFromInput(const MachInput &input,
                        mfem::Vector &vec,
                        bool deep_copy)
{
   auto *tmp = std::get<const mfem::Vector *>(input);
   if (deep_copy)
   {
      vec.SetSize(tmp->Size());
      vec = *tmp;
   }
   else
   {
      vec.NewDataAndSize(tmp->GetData(), tmp->Size());
   }
}

void setVectorFromInputs(const MachInputs &inputs,
                         const std::string &key,
                         mfem::Vector &vec,
                         bool deep_copy,
                         bool error_if_not_found)
{
   auto input = inputs.find(key);
   if (input != inputs.end())
   {
      setVectorFromInput(input->second, vec, deep_copy);
   }
   else if (error_if_not_found)
   {
      throw MachException("key = " + key + "not found in inputs!\n");
   }
}

}  // namespace mach
