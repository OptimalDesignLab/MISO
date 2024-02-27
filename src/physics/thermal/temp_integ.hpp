#ifndef MISO_TEMP_INTEGRATOR
#define MISO_TEMP_INTEGRATOR

#include "mfem.hpp"
//#include "solver.hpp"

namespace miso
{
class AggregateIntegratorNumerator : public mfem::NonlinearFormIntegrator
{
public:
   AggregateIntegratorNumerator(const double r, const mfem::Vector &m)
    : rho(r), max(m)
   { }

   double GetElementEnergy(const mfem::FiniteElement &el,
                           mfem::ElementTransformation &trans,
                           const mfem::Vector &elfun) override;

   // /// Computes dJdu, for the adjoint.
   // void AssembleElementVector(const mfem::FiniteElement &el,
   //                            mfem::ElementTransformation &trans,
   //                            const mfem::Vector &elfun,
   //                            mfem::Vector &elvect) override;
private:
   /// aggregation parameter rho
   const double rho;
   /// maximum allowable field value in each model region
   const mfem::Vector max;
};

class AggregateIntegratorDenominator : public mfem::NonlinearFormIntegrator
{
public:
   AggregateIntegratorDenominator(const double r, const mfem::Vector &m)
    : rho(r), max(m)
   { }

   double GetElementEnergy(const mfem::FiniteElement &el,
                           mfem::ElementTransformation &trans,
                           const mfem::Vector &elfun) override;

   // /// Computes dJdu, for the adjoint.
   // void AssembleElementVector(const mfem::FiniteElement &el,
   //                            mfem::ElementTransformation &trans,
   //                            const mfem::Vector &elfun,
   //                            mfem::Vector &elvect) override;
private:
   /// aggregation parameter rho
   const double rho;
   /// maximum allowable field value in each model region
   const mfem::Vector max;
};

/// Class that evaluates the aggregated temperature constraint
class AggregateIntegrator : public mfem::NonlinearFormIntegrator
{
public:
   /// Constructs a domain integrator that computes an induced aggregated
   /// temperature constraint based on the maximum value in the grid function
   AggregateIntegrator(const mfem::FiniteElementSpace *fe_space,
                       const double r,
                       const mfem::Vector &m)
    : fes(fe_space), rho(r), max(m)
   { }

   /// Overloaded, precomputes aggregate and denominator for use in adjoint
   AggregateIntegrator(const mfem::FiniteElementSpace *fe_space,
                       double r,
                       const mfem::Vector &m,
                       mfem::GridFunction *temp);

   /// Computes the induced functional estimate for aggregated temperature
   double GetIEAggregate(mfem::GridFunction *temp);

   /// Computes the induced functional estimate, need to call second constructor
   /// first
   double GetElementEnergy(const mfem::FiniteElement &el,
                           mfem::ElementTransformation &Trans,
                           const mfem::Vector &elfun) override;

   /// Computes dJdu, for the adjoint. Must call GetIEAggregate beforehand.
   void AssembleElementVector(const mfem::FiniteElement &el,
                              mfem::ElementTransformation &Trans,
                              const mfem::Vector &elfun,
                              mfem::Vector &elvect) override;

private:
   /// used to integrate over appropriate elements
   const mfem::FiniteElementSpace *fes;

   /// aggregation parameter rho
   const double rho;

   /// maximum temperature constraint (TODO: USE MULTIPLE MAXIMA, ONE FOR EACH
   /// MESH ATTRIBUTE)
   const mfem::Vector max;

   /// maximum temperature value
   double maxt{};

   // last computed output (for dJdU)
   double J_{};

   // last computed denom (for dJdU)
   double denom_{};

   // last computed state vector (for dJdu)
   mfem::GridFunction *temp_{};

#ifndef MFEM_THREAD_SAFE
   /// store the physical location of a node
   mfem::Vector x;
#endif
};

/// Class that integrates over temperature
class TempIntegrator : public mfem::NonlinearFormIntegrator
{
public:
   /// Constructs a domain integrator that computes the integral over
   /// temperature
   TempIntegrator(const mfem::FiniteElementSpace *fe_space) : fes(fe_space) { }

   /// Overloaded, precomputes for use in adjoint
   TempIntegrator(const mfem::FiniteElementSpace *fe_space,
                  mfem::GridFunction *temp);

   /// TODO: Turn this into a GetElementEnergy and GetFaceEnergy
   /// Computes the induced functional estimate for aggregated temperature
   double GetTemp(mfem::GridFunction *temp);

   /// Computes dJdu, for the adjoint
   void AssembleElementVector(const mfem::FiniteElement &el,
                              mfem::ElementTransformation &Trans,
                              const mfem::Vector &elfun,
                              mfem::Vector &elvect) override;

   /// Computes dJdu for the adjoint on the boundary
   void AssembleFaceVector(const mfem::FiniteElement &el1,
                           const mfem::FiniteElement &el2,
                           mfem::FaceElementTransformations &Trans,
                           const mfem::Vector &elfun,
                           mfem::Vector &elvect) override;

private:
   /// used to integrate over appropriate elements
   const mfem::FiniteElementSpace *fes;

   // last computed output (for dJdU)
   double J_{};

   // last computed denom (for dJdU)
   double denom_{};

   // last computed state vector (for dJdu)
   mfem::GridFunction *temp_;

#ifndef MFEM_THREAD_SAFE
   /// store the physical location of a node
   mfem::Vector x;
#endif
};

/// Class that evaluates the aggregated temperature constraint
/// (derivative with respect to mesh nodes)
class AggregateResIntegrator : public mfem::NonlinearFormIntegrator
{
public:
   /// Constructs a domain integrator that computes an induced aggregated
   /// temperature constraint based on the maximum value in the grid function
   AggregateResIntegrator(const mfem::FiniteElementSpace *fe_space,
                          const double r,
                          const mfem::Vector &m)
    : fes(fe_space), rho(r), max(m)
   { }

   /// Overloaded, precomputes aggregate and denominator for use in adjoint
   AggregateResIntegrator(const mfem::FiniteElementSpace *fe_space,
                          double r,
                          const mfem::Vector &m,
                          mfem::GridFunction *temp);

   /// Computes the induced functional estimate for aggregated temperature
   double GetIEAggregate(mfem::GridFunction *temp);

   /// Computes dJdx, for the adjoint. Must call GetIEAggregate beforehand.
   void AssembleElementVector(const mfem::FiniteElement &elx,
                              mfem::ElementTransformation &Trx,
                              const mfem::Vector &elfunx,
                              mfem::Vector &elvect) override;

private:
   /// used to integrate over appropriate elements
   const mfem::FiniteElementSpace *fes;

   /// aggregation parameter rho
   const double rho;

   /// maximum temperature constraint (TODO: USE MULTIPLE MAXIMA, ONE FOR EACH
   /// MESH ATTRIBUTE)
   const mfem::Vector max;

   /// maximum temperature value
   double maxt{};

   // last computed output (for dJdU)
   double J_{};

   // last computed denom (for dJdU)
   double denom_{};

   // last computed state vector (for dJdu)
   mfem::GridFunction *temp_{};

#ifndef MFEM_THREAD_SAFE
   /// store the physical location of a node
   mfem::Vector x;
#endif
};

/// Class that integrates over temperature
class TempResIntegrator : public mfem::NonlinearFormIntegrator
{
public:
   /// Constructs a domain integrator that computes the integral over
   /// temperature
   TempResIntegrator(const mfem::FiniteElementSpace *fe_space) : fes(fe_space)
   { }

   /// Overloaded, precomputes for use in adjoint
   TempResIntegrator(const mfem::FiniteElementSpace *fe_space,
                     mfem::GridFunction *temp);

   /// Computes the induced functional estimate for aggregated temperature
   double GetTemp(mfem::GridFunction *temp);

   /// Computes dJdu, for the adjoint
   void AssembleElementVector(const mfem::FiniteElement &elx,
                              mfem::ElementTransformation &Trx,
                              const mfem::Vector &elfunx,
                              mfem::Vector &elvect) override;

private:
   /// used to integrate over appropriate elements
   const mfem::FiniteElementSpace *fes;

   // last computed output (for dJdU)
   double J_{};

   // last computed denom (for dJdU)
   double denom_{};

   // last computed state vector (for dJdu)
   mfem::GridFunction *temp_;

#ifndef MFEM_THREAD_SAFE
   /// store the physical location of a node
   mfem::Vector x;
#endif
};

}  // namespace miso

#endif
