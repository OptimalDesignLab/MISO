#ifndef MACH_THERM_INTEG
#define MACH_THERM_INTEG

#include "mfem.hpp"

#include "coefficient.hpp"
#include "solver.hpp"

namespace mach
{

/// Integrator for (u, w) for H(grad) elements
class HeatMassIntegrator: public BilinearFormIntegrator
{
protected:
#ifndef MFEM_THREAD_SAFE
   Vector shape, te_shape;
#endif
   // TODO: Need two coefficients, density and cv

   Coefficient *Q;
   // PA extension
   Vector pa_data;
   const DofToQuad *maps;         ///< Not owned
   const GeometricFactors *geom;  ///< Not owned
   int dim, ne, nq, dofs1D, quad1D;

public:
   HeatMassIntegrator(const IntegrationRule *ir = NULL)
      : BilinearFormIntegrator(ir) { Q = NULL; maps = NULL; geom = NULL; }

   /// Construct a mass integrator with coefficient q
   HeatMassIntegrator(Coefficient &q, const IntegrationRule *ir = NULL)
      : BilinearFormIntegrator(ir), Q(&q) { maps = NULL; geom = NULL; }

   /** Given a particular Finite Element
       computes the element mass matrix elmat. */
   virtual void AssembleElementMatrix(const FiniteElement &el,
                                      ElementTransformation &Trans,
                                      DenseMatrix &elmat);

   virtual void AssemblePA(const FiniteElementSpace&);

   virtual void AddMultPA(const Vector&, Vector&) const;

   static const IntegrationRule &GetRule(const FiniteElement &trial_fe,
                                         const FiniteElement &test_fe,
                                         ElementTransformation &Trans);
};





} // namespace mach

#endif