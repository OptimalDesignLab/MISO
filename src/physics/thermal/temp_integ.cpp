
#include "temp_integ.hpp"

using namespace mfem;
using namespace std;

namespace mach
{

double AggregateIntegratorNumerator::GetElementEnergy(
   const FiniteElement &el, 
   ElementTransformation &Trans,
   const Vector &elfun)
{
   const int attr = Trans.Attribute;
   Vector shape(elfun.Size());

   const IntegrationRule *ir = &IntRules.Get(el.GetGeomType(),
                                             2 * el.GetOrder());

   double fun = 0.0;
   for (int i = 0; i < ir->GetNPoints(); ++i)
   {
      const IntegrationPoint &ip = ir->IntPoint(i);
      Trans.SetIntPoint(&ip);
      el.CalcShape(ip, shape);
      double g = (shape*elfun) / max(attr - 1);
      fun += ip.weight*Trans.Weight()*g*exp(rho*(g));
   }
   return fun;
}

double AggregateIntegratorDenominator::GetElementEnergy(
   const FiniteElement &el, 
   ElementTransformation &Trans,
   const Vector &elfun)
{
   const int attr = Trans.Attribute;
   Vector shape(elfun.Size());

   const IntegrationRule *ir = &IntRules.Get(el.GetGeomType(),
                                             2 * el.GetOrder());

   double fun = 0.0;
   for (int i = 0; i < ir->GetNPoints(); ++i)
   {
      const IntegrationPoint &ip = ir->IntPoint(i);
      Trans.SetIntPoint(&ip);
      el.CalcShape(ip, shape);
      double g = (shape*elfun) / max(attr - 1);
      fun += ip.weight*Trans.Weight()/exp(rho*(g));
   }
   return fun;
}



AggregateIntegrator::AggregateIntegrator(const FiniteElementSpace *fe_space,
                                         const double r,
                                         const Vector m,
                                         GridFunction *temp)       
   : fes(fe_space), rho(r), max(m)
{ 
   GetIEAggregate(temp);
}

double AggregateIntegrator::GetIEAggregate(GridFunction *temp)
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
      // const int dim = el->GetDim();
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

double AggregateIntegrator::GetElementEnergy(const FiniteElement &el, 
                                             ElementTransformation &Trans,
                                             const Vector &elfun)
{
   double Jpart = 0;
   // const int dof = el.GetDof();
   // const int dim = el.GetDim();
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

void AggregateIntegrator::AssembleElementVector(const FiniteElement &el, 
                                                ElementTransformation &Trans,
                                                const Vector &elfun,
                                                Vector &elvect)
{
   int dof = el.GetDof(); //, dim = el.GetDim();
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


TempIntegrator::TempIntegrator(const FiniteElementSpace *fe_space,
                               GridFunction *temp)       
   : fes(fe_space), temp_(temp)
{ 
   GetTemp(temp);
}

double TempIntegrator::GetTemp(GridFunction *temp)
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
      // const int dof = el->GetDof();
      // const int dim = el->GetDim();

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

void TempIntegrator::AssembleElementVector(const FiniteElement &el, 
                                           ElementTransformation &Trans,
                                           const Vector &elfun,
                                           Vector &elvect)
{
   int dof = el.GetDof(); //, dim = el.GetDim();
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

void TempIntegrator::AssembleFaceVector(const FiniteElement &el1, 
                                        const FiniteElement &el2, 
                                        FaceElementTransformations &Trans,
                                        const Vector &elfun,
                                        Vector &elvect)
{
   int dof = el1.GetDof(); //, dim = el1.GetDim();
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


AggregateResIntegrator::AggregateResIntegrator(
   const FiniteElementSpace *fe_space,
   const double r,
   const Vector m,
   GridFunction *temp)       
   : fes(fe_space), rho(r), max(m)
{ 
   GetIEAggregate(temp);
}

double AggregateResIntegrator::GetIEAggregate(GridFunction *temp)
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
      // const int dof = el->GetDof();
      // const int dim = el->GetDim();
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

void AggregateResIntegrator::AssembleElementVector(const FiniteElement &elx, 
                                                   ElementTransformation &Trx,
                                                   const Vector &elfunx,
                                                   Vector &elvect)
{
   /// get the proper element, transformation, and state vector
   Array<int> vdofs; Vector elfun; 
   int element = Trx.ElementNo;
   const FiniteElement *el = temp_->FESpace()->GetFE(element);
   ElementTransformation *Tr = temp_->FESpace()->GetElementTransformation(element);
   temp_->FESpace()->GetElementVDofs(element, vdofs);
   int order = 2*el->GetOrder() + Tr->OrderW();
   const IntegrationRule *ir = IntRule;
   if (ir == NULL)
   {
      ir = &IntRules.Get(el->GetGeomType(), order);
   }
   temp_->GetSubVector(vdofs, elfun);
   
   int dof = elx.GetDof(), dim = el->GetDim();
   elvect.SetSize(dof*dim);
   elvect = 0.0;
   DenseMatrix PointMat_bar(dim, dof);
   Vector DofVal(elfun.Size());

   // cast the ElementTransformation
   IsoparametricTransformation &isotrans =
   dynamic_cast<IsoparametricTransformation&>(*Tr);

   const int attr = Trx.Attribute;
   maxt = temp_->Max()/max(attr);

   for (int i = 0; i < ir->GetNPoints(); ++i)
   {
      const IntegrationPoint &ip = ir->IntPoint(i);
      Tr->SetIntPoint(&ip);
      el->CalcShape(ip, DofVal);
      
      PointMat_bar = 0.0;
      double val = (DofVal*elfun)/max(attr);
      
      double vexp = exp(rho*(val-maxt));
      double dnumer = ip.weight*val*vexp;
      dnumer -= J_*ip.weight*vexp;
      dnumer = dnumer/(denom_);

      isotrans.WeightRevDiff(PointMat_bar);
      PointMat_bar.Set(dnumer, PointMat_bar);

      for (int j = 0; j < dof ; ++j)
      {
         for (int d = 0; d < dim; ++d)
         {
            elvect(d*dof + j) += PointMat_bar(d,j);
         }
      }
   }
}

TempResIntegrator::TempResIntegrator( const mfem::FiniteElementSpace *fe_space,
                              mfem::GridFunction *temp)       
   : fes(fe_space), temp_(temp)
{ 
   GetTemp(temp);
}

double TempResIntegrator::GetTemp(mfem::GridFunction *temp)
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
      // const int dof = el->GetDof();
      // const int dim = el->GetDim();

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

void TempResIntegrator::AssembleElementVector(const mfem::FiniteElement &elx, 
               mfem::ElementTransformation &Trx,
               const mfem::Vector &elfunx, mfem::Vector &elvect)
{
   /// get the proper element, transformation, and state vector
   Array<int> vdofs; Vector elfun; 
   int element = Trx.ElementNo;
   const FiniteElement *el = temp_->FESpace()->GetFE(element);
   ElementTransformation *Tr = temp_->FESpace()->GetElementTransformation(element);
   temp_->FESpace()->GetElementVDofs(element, vdofs);
   int order = 2*el->GetOrder() + Tr->OrderW();
   const IntegrationRule *ir = IntRule;
   if (ir == NULL)
   {
      ir = &IntRules.Get(el->GetGeomType(), order);
   }
   temp_->GetSubVector(vdofs, elfun);
   
   int dof = elx.GetDof(), dim = el->GetDim();
   elvect.SetSize(dof*dim);
   elvect = 0.0;
   DenseMatrix PointMat_bar(dim, dof);
   Vector DofVal(elfun.Size());

   // cast the ElementTransformation
   IsoparametricTransformation &isotrans =
   dynamic_cast<IsoparametricTransformation&>(*Tr);

   for (int i = 0; i < ir->GetNPoints(); ++i)
   {
      const IntegrationPoint &ip = ir->IntPoint(i);
      Tr->SetIntPoint(&ip);
      el->CalcShape(ip, DofVal);
      
      PointMat_bar = 0.0;
      double val = (DofVal*elfun);
      double dnumer = val*ip.weight - J_*ip.weight;
      dnumer = dnumer/denom_; 

      isotrans.WeightRevDiff(PointMat_bar);
      PointMat_bar.Set(dnumer, PointMat_bar);

      for (int j = 0; j < dof ; ++j)
      {
         for (int d = 0; d < dim; ++d)
         {
            elvect(d*dof + j) += PointMat_bar(d,j);
         }
      }
   }
}

} // namespace mach
