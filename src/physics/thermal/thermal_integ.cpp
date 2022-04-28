#include "mfem.hpp"

#include "mach_input.hpp"

#include "thermal_integ.hpp"

namespace mach
{

void ConvectionBCIntegrator::AssembleFaceVector(
    const mfem::FiniteElement &el1,
    const mfem::FiniteElement &el2,
    mfem::FaceElementTransformations &trans,
    const mfem::Vector &elfun,
    mfem::Vector &elvect)
{
   int ndof = el1.GetDof();

#ifdef MFEM_THREAD_SAFE
   mfem::Vector shape;
#endif
   shape.SetSize(ndof);
   elvect.SetSize(ndof);

   const auto *ir = IntRule;
   if (ir == nullptr)
   {
      int order = el1.GetOrder() + el2.GetOrder() + trans.OrderW();
      ir = &mfem::IntRules.Get(el1.GetGeomType(), order);
   }

   elvect = 0.0;
   for (int i = 0; i < ir->GetNPoints(); i++)
   {
      // Set the integration point in the face and the neighboring element
      const auto &ip = ir->IntPoint(i);
      trans.SetAllIntPoints(&ip);

      const double w = alpha * ip.weight * trans.Face->Weight();

      // Access the neighboring element's integration point
      const auto &eip = trans.GetElement1IntPoint();
      el1.CalcShape(eip, shape);

      const double val = h * ((elfun * shape) - theta_f);

      add(elvect, w * val, shape, elvect);
   }
}

void ConvectionBCIntegrator::AssembleFaceGrad(
    const mfem::FiniteElement &el1,
    const mfem::FiniteElement &el2,
    mfem::FaceElementTransformations &trans,
    const mfem::Vector &elfun,
    mfem::DenseMatrix &elmat)
{
   int ndof = el1.GetDof();

#ifdef MFEM_THREAD_SAFE
   mfem::Vector shape;
#endif
   shape.SetSize(ndof);
   elmat.SetSize(ndof);

   const auto *ir = IntRule;
   if (ir == nullptr)
   {
      int order = el1.GetOrder() + el2.GetOrder() + trans.OrderW();
      ir = &mfem::IntRules.Get(el1.GetGeomType(), order);
   }

   elmat = 0.0;
   for (int i = 0; i < ir->GetNPoints(); i++)
   {
      // Set the integration point in the face and the neighboring element
      const auto &ip = ir->IntPoint(i);
      trans.SetAllIntPoints(&ip);

      const double w = alpha * ip.weight * trans.Face->Weight();

      // Access the neighboring element's integration point
      const auto &eip = trans.GetElement1IntPoint();
      el1.CalcShape(eip, shape);

      AddMult_a_VVt(w * h, shape, elmat);
   }
}

}  // namespace mach
