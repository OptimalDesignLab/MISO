#include "utils.hpp"
#include "mach_input.hpp"

namespace mach
{

double* MachInput::getField() const
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

} // namespace mach
