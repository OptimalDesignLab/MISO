
#include "temp_integ.hpp"

using namespace mfem;
using namespace std;

namespace mach
{


double AggregateIntegrator::GetIEAggregate(GridFunType *temp)
{
   cout.flush();
   Array<int> dofs;
   ElementTransformation *eltrans;

   double numer = 0;
   double denom = 0;

   maxt = temp->Max()/max;

   // loop through elements
   // TODO: USE MULTIPLE MAXIMA, ONE FOR EACH MESH ATTRIBUTE)
   for (int j = 0; j < fes->GetNE(); j++)
   {
      fes->GetElementDofs(j, dofs);
      eltrans = fes->GetElementTransformation(j);
      const FiniteElement *el = fes->GetFE(j);
      const int dim = el->GetDim();
      x.SetSize(dim);

      // loop through nodes
      for (int i = 0; i < el->GetDof(); ++i)
      {
         const IntegrationPoint &ip = el->GetNodes().IntPoint(i);
         eltrans->SetIntPoint(&ip);
         double val = temp->GetValue(j, ip)/max;

         numer += eltrans->Weight()*val*exp(rho*(val - maxt));

         denom += eltrans->Weight()*exp(rho*(val - maxt));
      }
   }
   //std::cout << "max temp: " << max << endl;

   return numer/denom;
}


} // namespace mach
