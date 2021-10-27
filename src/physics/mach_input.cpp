#include "utils.hpp"
#include "mach_input.hpp"

namespace mach
{
double *MachInput::getField() const
{
   if (active != Type::Field)
   {
      throw MachException("Input type is not a field!\n");
   }
   else
   {
      return input.field;
   }
}

double MachInput::getValue() const
{
   if (active != Type::Value)
   {
      throw MachException("Input type is not a value!\n");
   }
   else
   {
      return input.value;
   }
}

void setValueFromInputs(const MachInputs &inputs,
                        const std::string &key,
                        double &value,
                        bool error_if_not_found)
{
   auto it = inputs.find(key);
   if (it != inputs.end())
   {
      value = it->second.getValue();
   }
   else if (error_if_not_found)
   {
      throw MachException("key = " + key + "not found in inputs!\n");
   }
}

void setFieldFromInputs(const MachInputs &inputs,
                        const std::string &key,
                        double *field,
                        bool error_if_not_found)
{
   auto it = inputs.find(key);
   if (it != inputs.end())
   {
      field = it->second.getField();
   }
   else if (error_if_not_found)
   {
      throw MachException("key = " + key + "not found in inputs!\n");
   }
}

}  // namespace mach
