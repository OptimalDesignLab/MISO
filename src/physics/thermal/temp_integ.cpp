
#include "temp_integ.hpp"

using namespace mfem;
using namespace std;

namespace mach
{

AggregateIntegrator::AggregateIntegrator(
                              const mfem::FiniteElementSpace *fe_space,
                              const double r,
                              const mfem::Vector m,
                              mfem::GridFunction *temp)       
   : fes(fe_space), rho(r), max(m)
{ 
   GetIEAggregate(temp);
}

double AggregateIntegrator::GetIEAggregate(mfem::GridFunction *temp)
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
      const int dof = el->GetDof();
      const int dim = el->GetDim();
      const int attr = fes->GetAttribute(j);

      maxt = temp->Max()/max(attr);

      const IntegrationRule *ir = &IntRules.Get(el->GetGeomType(), 2 * el->GetOrder());

      // loop through nodes
      for (int i = 0; i < ir->GetNPoints(); ++i)
      {
         const IntegrationPoint &ip = ir->IntPoint(i);
         eltrans->SetIntPoint(&ip);
         double val = temp->GetValue(j, ip)/max(attr);

         numer += ip.weight*eltrans->Weight()*val*exp(rho*(val - maxt));

         denom += ip.weight*eltrans->Weight()*exp(rho*(val - maxt));
      }
   }
   //std::cout << "max temp: " << max << endl;

   J_ = numer/denom;
   denom_ = denom;
   temp_ = temp;

   return J_;
}

double AggregateIntegrator::GetElementEnergy(const mfem::FiniteElement &el, 
               mfem::ElementTransformation &Trans,
               const mfem::Vector &elfun)
{
   double Jpart = 0;
   const int dof = el.GetDof();
   const int dim = el.GetDim();
   const int attr = Trans.Attribute;
   Vector DofVal(elfun.Size());
   maxt = temp_->Max()/max(attr);
   const IntegrationRule *ir = &IntRules.Get(el.GetGeomType(), 2 * el.GetOrder());
   // loop through nodes
   for (int i = 0; i < ir->GetNPoints(); ++i)
   {
      const IntegrationPoint &ip = ir->IntPoint(i);
      Trans.SetIntPoint(&ip);
      el.CalcShape(ip, DofVal);
      double val = (DofVal*elfun)/max(attr);
      Jpart += ip.weight*Trans.Weight()*val*exp(rho*(val - maxt));
   }

   return Jpart/denom_;
}

void AggregateIntegrator::AssembleElementVector(const mfem::FiniteElement &el, 
               mfem::ElementTransformation &Trans,
               const mfem::Vector &elfun, mfem::Vector &elvect)
{
   int dof = el.GetDof(), dim = el.GetDim();
   elvect.SetSize(dof);
   elvect = 0.0;
   Vector DofVal(elfun.Size());

   const int attr = Trans.Attribute;
   maxt = temp_->Max()/max(attr);

   const IntegrationRule *ir = &IntRules.Get(el.GetGeomType(), 2*el.GetOrder());

   for (int i = 0; i < ir->GetNPoints(); ++i)
   {
      const IntegrationPoint &ip = ir->IntPoint(i);
      Trans.SetIntPoint(&ip);
      el.CalcShape(ip, DofVal);
      double val = (DofVal*elfun)/max(attr);
      
      double vexp = exp(rho*(val-maxt));
      double dnumer = Trans.Weight()*(1 + rho*val - J_*rho)*vexp;

      add(elvect, ip.weight*dnumer/(denom_*max(attr)), DofVal, elvect);
   }
}


TempIntegrator::TempIntegrator( const mfem::FiniteElementSpace *fe_space,
                              mfem::GridFunction *temp)       
   : fes(fe_space), temp_(temp)
{ 
   //GetTemp(temp);
}

double TempIntegrator::GetTemp(mfem::GridFunction *temp)
{
   cout.flush();
   Array<int> dofs;
   ElementTransformation *eltrans;

   double numer = 0;
   double denom = 0;

   // loop through elements
   for (int j = 0; j < fes->GetNE(); j++)
   {
      fes->GetElementDofs(j, dofs);
      eltrans = fes->GetElementTransformation(j);
      const FiniteElement *el = fes->GetFE(j);
      const int dof = el->GetDof();
      const int dim = el->GetDim();

      const IntegrationRule *ir = &IntRules.Get(el->GetGeomType(), 2 * el->GetOrder());

      // loop through nodes
      for (int i = 0; i < ir->GetNPoints(); ++i)
      {
         const IntegrationPoint &ip = ir->IntPoint(i);
         eltrans->SetIntPoint(&ip);
         double val = temp->GetValue(j, ip);

         numer += ip.weight*eltrans->Weight()*val;

         denom += ip.weight*eltrans->Weight();
      }
   }
   //std::cout << "max temp: " << max << endl;

   J_ = numer/denom;
   denom_ = denom;
   temp_ = temp;

   return J_;
}

void TempIntegrator::AssembleElementVector(const mfem::FiniteElement &el, 
               mfem::ElementTransformation &Trans,
               const mfem::Vector &elfun, mfem::Vector &elvect)
{
   int dof = el.GetDof(), dim = el.GetDim();
   elvect.SetSize(dof);
   elvect = 0.0;
   Vector DofVal(elfun.Size());

   const IntegrationRule *ir = &IntRules.Get(el.GetGeomType(), 2*el.GetOrder());

   for (int i = 0; i < ir->GetNPoints(); ++i)
   {
      const IntegrationPoint &ip = ir->IntPoint(i);
      Trans.SetIntPoint(&ip);
      el.CalcShape(ip, DofVal);

      add(elvect, ip.weight*Trans.Weight()/(denom_), DofVal, elvect);
   }
}

void TempIntegrator::AssembleFaceVector(const mfem::FiniteElement &el1, 
               const mfem::FiniteElement &el2, 
               mfem::FaceElementTransformations &Trans,
               const mfem::Vector &elfun, mfem::Vector &elvect)
{
   int dof = el1.GetDof(), dim = el1.GetDim();
   elvect.SetSize(dof);
   elvect = 0.0;
   denom_ = 1.0; //area of face, use for testing only
   Vector DofVal(elfun.Size());

   const IntegrationRule *ir = &IntRules.Get(Trans.FaceGeom, 2*el1.GetOrder());

   for (int i = 0; i < ir->GetNPoints(); ++i)
   {
      const IntegrationPoint &ip = ir->IntPoint(i);
      IntegrationPoint eip;
      Trans.Loc1.Transform(ip, eip);
      Trans.Face->SetIntPoint(&ip);
      el1.CalcShape(eip, DofVal);

      add(elvect, ip.weight*Trans.Face->Weight()/(denom_), DofVal, elvect);
   }
}


} // namespace mach
