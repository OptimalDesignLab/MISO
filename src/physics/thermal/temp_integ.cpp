
#include "temp_integ.hpp"

using namespace mfem;
using namespace std;

namespace mach
{

AggregateIntegrator::AggregateIntegrator(
                              const mfem::FiniteElementSpace *fe_space,
                              const double r,
                              const mfem::Vector m,
                              GridFunType *temp)       
   : fes(fe_space), rho(r), max(m)
{ 
   GetIEAggregate(temp);
}

double AggregateIntegrator::GetIEAggregate(GridFunType *temp)
{
   cout.flush();
   Array<int> dofs;
   ElementTransformation *eltrans;

   double numer = 0;
   double denom = 0;

   // loop through elements
   // TODO: USE MULTIPLE MAXIMA, ONE FOR EACH MESH ATTRIBUTE)
   for (int j = 0; j < fes->GetNE(); j++)
   {
      fes->GetElementDofs(j, dofs);
      eltrans = fes->GetElementTransformation(j);
      const FiniteElement *el = fes->GetFE(j);
      const int dim = el->GetDim();
      const int attr = fes->GetAttribute(j);

      maxt = temp->Max()/max(attr);

      // loop through nodes
      for (int i = 0; i < el->GetDof(); ++i)
      {
         const IntegrationPoint &ip = el->GetNodes().IntPoint(i);
         eltrans->SetIntPoint(&ip);
         double val = temp->GetValue(j, ip)/max(attr);

         numer += eltrans->Weight()*val*exp(rho*(val - maxt));

         denom += eltrans->Weight()*exp(rho*(val - maxt));
      }
   }
   //std::cout << "max temp: " << max << endl;

   J_ = numer/denom;
   denom_ = denom;
   temp_ = temp;

   return J_;
}

void AggregateIntegrator::AssembleElementVector(const mfem::FiniteElement &el, 
               mfem::ElementTransformation &Trans,
               const mfem::Vector &elfun, mfem::Vector &elvect)
{
   int dof = el.GetDof(), dim = el.GetDim();
   elvect.SetSize(dof);

   const int attr = Trans.Attribute;
   maxt = temp_->Max()/max(attr);

   for (int i = 0; i < el.GetDof(); ++i)
   {
      const IntegrationPoint &ip = el.GetNodes().IntPoint(i);
      Trans.SetIntPoint(&ip);
      double val = elfun(i)/max(attr);
      
      double vexp = exp(rho*(val-maxt));
      double dnumer = Trans.Weight()*(1 + rho*val - J_*rho)*vexp;

      elvect(i) = dnumer/(denom_*max(attr));
   }
}


} // namespace mach
