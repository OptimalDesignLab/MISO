#ifndef MACH_INPUT
#define MACH_INPUT

#include <string>
#include <unordered_map>

#include "mfem.hpp"

namespace mach
{
/// Class that represents possible input variabels used when evaluating a mach
/// solver. Currently supported input types include either a field variable
/// (array) or a scalar variable. The class is backed by a union that stores
/// either the field or the scalar, and uses an enum to keep track of what type
/// the union is holding.
/// This class is meant to be similar in spirit to C++17's std::variant, but
/// uses only C++11 features.
// class MachInput
// {
// public:
//    MachInput(double *field) : input(field), active(Type::Field) { }
//    MachInput(double value) : input(value), active(Type::Value) { }

//    /// access the input data
//    double *getField() const;
//    double getValue() const;

//    /// allows querying to determine if the input is a field or a scalar
//    bool isField() const { return active == Type::Field; }
//    bool isValue() const { return active == Type::Value; }

// private:
//    union Input
//    {
//       Input(double *field) : field(field) { }
//       Input(double value) : value(value) { }

//       double *field;
//       double value;
//    };

//    enum Type
//    {
//       Field,
//       Value
//    };

//    Input input;
//    Type active;
// };

/// Helper class that gives mfem::Vector reference semantics allowing
/// shallow-copy emplacement into MachInput
struct InputVector final
{
   InputVector(double *data, int size) : data(data), size(size) { }
   InputVector(const mfem::Vector &v) : data(v.GetData()), size(v.Size()) { }
   operator mfem::Vector() const { return mfem::Vector(data, size); }

   double *data;
   int size;
};

/// Convenient alias representing the possible input types for mach solvers
using MachInput = std::variant<double, InputVector>;

/// Convenient shorthand for a map of inputs since each input must be named
using MachInputs = std::unordered_map<std::string, MachInput>;

/// Helper function that scans a `MachInput` and sets a value
/// \param[in] input - given MachInput holding a value
/// \param[out] value - value to be set using from value stored in @a input
void setValueFromInput(const MachInput &input, double &value);

/// Helper function that scans a `MachInput` for a given `key` and sets value
/// \param[in] inputs - map of strings to MachInputs
/// \param[in] key - value to look for in `inputs`
/// \param[out] value - if `key` is found, value is set to `inputs.at(key)`
/// \param[in] error_if_not_found - if true, and `key` not found, raises
void setValueFromInputs(const MachInputs &inputs,
                        const std::string &key,
                        double &value,
                        bool error_if_not_found = false);

/// Helper function that scans a `MachInput` and sets Vector
/// \param[in] input - given MachInput holding a vector
/// \param[out] vec - vector to be set using from value stored in @a input
/// \param[in] deep_copy - if true, deep copy the input vector into @a vec
void setVectorFromInput(const MachInput &input,
                        mfem::Vector &vec,
                        bool deep_copy = false);

/// Helper function that scans a `MachInput` for a given `key` and sets Vector
/// \param[in] inputs - map of strings to MachInputs
/// \param[in] key - value to look for in `inputs`
/// \param[out] vec - if `key` is found, vector is set using `inputs.at(key)`
/// \param[in] deep_copy - if true, deep copy the input vector into @a vec
/// \param[in] error_if_not_found - if true, and `key` not found, raises
void setVectorFromInputs(const MachInputs &inputs,
                         const std::string &key,
                         mfem::Vector &vec,
                         bool deep_copy = false,
                         bool error_if_not_found = false);

}  // namespace mach

#endif
