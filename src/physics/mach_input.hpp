#ifndef MACH_INPUT
#define MACH_INPUT

#include <string>
#include <unordered_map>

namespace mach
{

class MachInput
{
public:
   MachInput(double *field)
   : input(field), active(Type::Field)
   { }
   MachInput(double value)
   : input(value), active(Type::Value)
   { }

   ~MachInput() = default;

   double* getField() const;
   double getValue() const;

   bool isField() const { return active == Type::Field; }
   bool isValue() const { return active == Type::Value; }

private:
   union Input
   {
      Input(double* field)
      : field(field)
      { }
      Input(double value)
      : value(value)
      { }

      double *field;
      double value;
   };

   enum Type {Field, Value};

   const Input input;
   const Type active;
};

using MachInputs = std::unordered_map<std::string, MachInput>;

} // namespace mach

#endif
