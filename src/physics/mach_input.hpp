#ifndef MACH_INPUT
#define MACH_INPUT

#include <string>
#include <unordered_map>

namespace mach
{
/// Class that represents possible input variabels used when evaluating a mach
/// solver. Currently supported input types include either a field variable
/// (array) or a scalar variable. The class is backed by a union that stores
/// either the field or the scalar, and uses an enum to keep track of what type
/// the union is holding.
/// This class is meant to be similar in spirit to C++17's std::variant, but
/// uses only C++11 features.
class MachInput
{
public:
   MachInput(double *field) : input(field), active(Type::Field) { }
   MachInput(double value) : input(value), active(Type::Value) { }

   /// access the input data
   double *getField() const;
   double getValue() const;

   /// allows querying to determine if the input is a field or a scalar
   bool isField() const { return active == Type::Field; }
   bool isValue() const { return active == Type::Value; }

private:
   union Input
   {
      Input(double *field) : field(field) { }
      Input(double value) : value(value) { }

      double *field;
      double value;
   };

   enum Type
   {
      Field,
      Value
   };

   Input input;
   Type active;
};

/// Convenient shorthand for a map of inputs since each input must be named
using MachInputs = std::unordered_map<std::string, MachInput>;

/// Helper function that scans a `MachInput` for a given `key` and sets value 
/// \param[in] inputs - map of strings to MachInputs 
/// \param[in] key - value to look for in `inputs`
/// \param[out] value - if `key` is found, value is set to `inputs.at(key)`
/// \param[in] error_if_not_found - if true, and `key` not found, raises
/// \note If `key` is not found, `value` is left unchanged.
void setValueFromInputs(const MachInputs &inputs, const std::string &key,
                        double &value, bool error_if_not_found=false);

/// Helper function that scans a `MachInput` for a given `key` and sets field 
/// \param[in] inputs - map of strings to MachInputs 
/// \param[in] key - value to look for in `inputs`
/// \param[out] value - if `key` is found, field is set to `inputs.at(key)`
/// \param[in] error_if_not_found - if true, and `key` not found, raises
/// \note If `key` is not found, `field` is left unchanged.
void setFieldFromInputs(const MachInputs &inputs, const std::string &key, 
                        double *field, bool error_if_not_found=false);

}  // namespace mach

#endif
