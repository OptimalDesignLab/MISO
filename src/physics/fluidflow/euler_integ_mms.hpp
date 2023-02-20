#ifndef INVISCID_MMS_INTEG
#define INVISCID_MMS_INTEG

#include "adept.h"
#include "mfem.hpp"

#include "mms_integ.hpp"
#include "euler_fluxes.hpp"
#include "mach_input.hpp"

namespace mach
{
/// Source-term integrator for a 2D/3D Inviscid MMS problem
/// \note For details on the MMS problem, see the file viscous_mms.py
class InviscidMMSIntegrator
 : public MMSIntegrator<InviscidMMSIntegrator>
{
public:
   /// Construct an integrator for a 2D/3D Inviscid MMS source
   /// \param[in] a - used to move residual to lhs (1.0) or rhs(-1.0)
   InviscidMMSIntegrator(double a = -1.0, int dim_num = 2)
    : MMSIntegrator<InviscidMMSIntegrator>(dim_num+2, a), dim(dim_num)
   { }

   /// Computes the MMS source term at a give point
   /// \param[in] x - spatial location at which to evaluate the source
   /// \param[out] src - source term evaluated at `x`
   void calcSource(const mfem::Vector &x, mfem::Vector &src) const
   {
      calcInviscidMMS<double>(dim, x.GetData(), src.GetData());
   }

private:
   /// Dimension of the problem
   int dim;
};
}

#endif INVISCID_MMS_INTEG