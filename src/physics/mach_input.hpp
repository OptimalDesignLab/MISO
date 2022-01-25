#ifndef MACH_INPUT
#define MACH_INPUT

#include <string>
#include <unordered_map>
#include <variant>

#include "mfem.hpp"
#include "nlohmann/json.hpp"

namespace mach
{
/// Helper class that gives mfem::Vector reference semantics allowing
/// shallow-copy emplacement into MachInput
struct InputVector final
{
   InputVector(double *data, int size) : data(data), size(size) { }
   InputVector(const mfem::Vector &v) : data(v.GetData()), size(v.Size()) { }
   operator mfem::Vector() const { return {data, size}; }

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

template <typename T>
void setInputs(T & /*unused*/, const MachInputs & /*unused*/)
{ }

template <typename T>
void setOptions(T & /*unused*/, const nlohmann::json & /*unused*/)
{ }

}  // namespace mach

#endif
