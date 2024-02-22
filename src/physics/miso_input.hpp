#ifndef MISO_INPUT
#define MISO_INPUT

#include <string>
#include <unordered_map>
#include <variant>

#include "mfem.hpp"
#include "nlohmann/json.hpp"

namespace miso
{
/// Helper class that gives mfem::Vector reference semantics allowing
/// shallow-copy emplacement into MISOInput
struct InputVector final
{
   InputVector(double *data, int size) : data(data), size(size) { }
   InputVector(const mfem::Vector &v) : data(v.GetData()), size(v.Size()) { }
   operator mfem::Vector() const { return {data, size}; }

   double *data;
   int size;
};

/// Convenient alias representing the possible input types for MISO solvers
using MISOInput = std::variant<double, InputVector>;

/// Convenient shorthand for a map of inputs since each input must be named
using MISOInputs = std::unordered_map<std::string, MISOInput>;

/// Helper function that scans a `MISOInput` and sets a value
/// \param[in] input - given MISOInput holding a value
/// \param[out] value - value to be set using from value stored in @a input
void setValueFromInput(const MISOInput &input, double &value);

/// Helper function that scans a `MISOInput` for a given `key` and sets value
/// \param[in] inputs - map of strings to MISOInputs
/// \param[in] key - value to look for in `inputs`
/// \param[out] value - if `key` is found, value is set to `inputs.at(key)`
/// \param[in] error_if_not_found - if true, and `key` not found, raises
void setValueFromInputs(const MISOInputs &inputs,
                        const std::string &key,
                        double &value,
                        bool error_if_not_found = false);

/// Helper function that scans a `MISOInput` and sets Vector
/// \param[in] input - given MISOInput holding a vector
/// \param[out] vec - vector to be set using from value stored in @a input
/// \param[in] deep_copy - if true, deep copy the input vector into @a vec
void setVectorFromInput(const MISOInput &input,
                        mfem::Vector &vec,
                        bool deep_copy = false);

/// Helper function that scans a `MISOInput` for a given `key` and sets Vector
/// \param[in] inputs - map of strings to MISOInputs
/// \param[in] key - value to look for in `inputs`
/// \param[out] vec - if `key` is found, vector is set using `inputs.at(key)`
/// \param[in] deep_copy - if true, deep copy the input vector into @a vec
/// \param[in] error_if_not_found - if true, and `key` not found, raises
void setVectorFromInputs(const MISOInputs &inputs,
                         const std::string &key,
                         mfem::Vector &vec,
                         bool deep_copy = false,
                         bool error_if_not_found = false);

template <typename T>
void setInputs(T & /*unused*/, const MISOInputs & /*unused*/)
{ }

template <typename T>
void setOptions(T & /*unused*/, const nlohmann::json & /*unused*/)
{ }

}  // namespace miso

#endif
