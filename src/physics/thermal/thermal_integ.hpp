#ifndef MACH_THERMAL_INTEG
#define MACH_THERMAL_INTEG

#include "mfem.hpp"

#include "mach_input.hpp"
#include "mfem_common_integ.hpp"

namespace mach
{
class ConvectionBCIntegrator : public mfem::NonlinearFormIntegrator
{
public:
   friend void setInputs(ConvectionBCIntegrator &integ,
                         const MachInputs &inputs)
   {
      setValueFromInputs(inputs, "h", integ.h);
      setValueFromInputs(inputs, "fluid_temp", integ.theta_f);
   }

   void AssembleFaceVector(const mfem::FiniteElement &el1,
                           const mfem::FiniteElement &el2,
                           mfem::FaceElementTransformations &trans,
                           const mfem::Vector &elfun,
                           mfem::Vector &elvect) override;

   void AssembleFaceGrad(const mfem::FiniteElement &el1,
                         const mfem::FiniteElement &el2,
                         mfem::FaceElementTransformations &trans,
                         const mfem::Vector &elfun,
                         mfem::DenseMatrix &elmat) override;

   ConvectionBCIntegrator(double alpha = 1.0) : alpha(alpha) { }

private:
   /// scales the terms; can be used to move to rhs/lhs
   double alpha;
   /// Convection heat transfer coefficient
   double h = 1.0;
   /// Fluid temperature
   double theta_f = 0.0;
#ifndef MFEM_THREAD_SAFE
   mfem::Vector shape;
#endif
};

}  // namespace mach

#endif