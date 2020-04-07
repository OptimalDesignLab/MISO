#ifndef MACH_TEMP_INTEGRATOR
#define MACH_TEMP_INTEGRATOR

#include "mfem.hpp"
#include "solver.hpp"

namespace mach
{
/// Class that evaluates the aggregated temperature constraint
class AggregateIntegrator : public mfem::NonlinearFormIntegrator
{
public:
   /// Constructs a domain integrator that computes an induced aggregated 
   /// temperature constraint based on the maximum value in the grid function
   AggregateIntegrator(const mfem::FiniteElementSpace *fe_space,
                              const double r,
                              const mfem::Vector m)
       : fes(fe_space), rho(r), max(m) { }
   
   /// Overloaded, precomputes aggregate and denominator for use in adjoint
   AggregateIntegrator(const mfem::FiniteElementSpace *fe_space,
                              const double r,
                              const mfem::Vector m,
                              GridFunType *temp);

   /// Computes the induced functional estimate for aggregated temperature
	double GetIEAggregate(GridFunType *temp);

   /// Computed dJdu, for the adjoint. Must call GetIEAggregate beforehand.
   virtual void AssembleElementVector(const mfem::FiniteElement &el, 
               mfem::ElementTransformation &Trans,
               const mfem::Vector &elfun, mfem::Vector &elvect);
private: 

   /// used to integrate over appropriate elements
   const mfem::FiniteElementSpace *fes;

   /// aggregation parameter rho
   const double rho;

   /// maximum temperature constraint (TODO: USE MULTIPLE MAXIMA, ONE FOR EACH MESH ATTRIBUTE)
   const mfem::Vector max;

   /// maximum temperature value
   double maxt;

   // last computed output (for dJdU)
   double J_;

   // last computed denom (for dJdU)
   double denom_;

   // last computed state vector (for dJdu)
   GridFunType *temp_;

#ifndef MFEM_THREAD_SAFE
   /// store the physical location of a node
   mfem::Vector x;
#endif
};

} // namespace mach

#endif
