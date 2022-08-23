#ifndef MACH_RELAXED_NEWTON
#define MACH_RELAXED_NEWTON

#include <memory>

#include "mfem.hpp"
#include "nlohmann/json.hpp"

#include "linesearch.hpp"

namespace mach
{
/// Newton's method for solving F(x) = b augmented with a linesearch
class RelaxedNewton : public mfem::NewtonSolver
{
public:
   RelaxedNewton(MPI_Comm comm, const nlohmann::json &options);

   double ComputeScalingFactor(const mfem::Vector &x,
                               const mfem::Vector &b) const override;

private:
   mutable mfem::Vector scratch;

   std::unique_ptr<LineSearch> ls;
};

}  // namespace mach

#endif
