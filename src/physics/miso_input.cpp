#include "utils.hpp"
#include "miso_input.hpp"

namespace miso
{
double *MISOInput::getField() const
{
   if (active != Type::Field)
   {
      throw MISOException("Input type is not a field!\n");
   }
   else
   {
      return input.field;
   }
}

double MISOInput::getValue() const
{
   if (active != Type::Value)
   {
      throw MISOException("Input type is not a value!\n");
   }
   else
   {
      return input.value;
   }
}

}  // namespace miso
