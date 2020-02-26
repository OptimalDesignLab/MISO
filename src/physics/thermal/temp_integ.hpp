#ifndef MACH_TEMP_INTEGRATOR
#define MACH_TEMP_INTEGRATOR

#include "mfem.hpp"
#include "solver.hpp"

namespace mach
{

/// Class that evaluates the aggregated temperature constraint
class AggregateIntegrator 
{
public:
   /// Constructs a domain integrator that computes an induced aggregated 
   /// temperature constraint based on the maximum value in the grid function
   /// \param[in] fe_coll - used to determine the face elements
   AggregateIntegrator(const mfem::FiniteElementSpace *fe_space,
                              const double r,
                              const mfem::Vector m)
       : fes(fe_space), rho(r), max(m) { }
   
   /// Computes the induced functional estimate for aggregated temperature
	double GetIEAggregate(GridFunType *temp);

private: 

   /// used to integrate over appropriate elements
   const mfem::FiniteElementSpace *fes;

   /// aggregation parameter rho
   const double rho;

   /// maximum temperature constraint (TODO: USE MULTIPLE MAXIMA, ONE FOR EACH MESH ATTRIBUTE)
   const mfem::Vector max;

   /// maximum temperature value
   double maxt;

#ifndef MFEM_THREAD_SAFE
   /// store the physical location of a node
   mfem::Vector x;
#endif
};

} // namespace mach

#endif
