#ifndef MACH_LINESEARCH
#define MACH_LINESEARCH

#include <functional>

#include "mfem.hpp"

namespace mach
{
class LineSearch
{
public:
   virtual double search(const std::function<double(double)> &phi,
                         double phi0,
                         double dphi0,
                         double alpha) = 0;

   virtual ~LineSearch() = default;
};

class BacktrackingLineSearch : public LineSearch
{
public:
   double search(const std::function<double(double)> &phi,
                 double phi0,
                 double dphi0,
                 double alpha) override;

   double mu = 1e-4;
   double rho_hi = 0.9;
   double rho_lo = 0.1;
   int interp_order = 3;
   int max_iter = 10;
};

/// Functor class for \phi, the 1D function linesearch methods try to minimize
class Phi
{
public:
   Phi(std::function<void(const mfem::Vector &x, mfem::Vector &res)> calcRes,
       const mfem::Vector &state,
       const mfem::Vector &descent_dir,
       mfem::Vector &residual,
       mfem::Operator &jac);

   double operator()(double alpha);

private:
   std::function<void(const mfem::Vector &x, mfem::Vector &res)> calcRes;
   const mfem::Vector &state;
   const mfem::Vector &descent_dir;
   mfem::Vector scratch;
   mfem::Vector &residual;

public:
   const double phi0;
   const double dphi0;
};

}  // namespace mach

#endif
