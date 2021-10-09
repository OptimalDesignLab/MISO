#include "parcentgridfunc.hpp"
#include "galer_diff.hpp"

using namespace mfem;

namespace mfem
{

ParCentGridFunction::ParCentGridFunction(ParFiniteElementSpace *pf, int pr)
   : ParGridFunction(pf)
{
   proc = pr;
   t_vec.SetSize(pf->GetTrueVSize());
   //SetSize(pf->GetVDim() * dynamic_cast<ParGDSpace*>(pf)->GetNDofs());
}

void ParCentGridFunction::ProjectCoefficient(VectorCoefficient &coeff)
{
   int vdim = pfes->GetVDim();
   Array<int> vdofs(vdim);
   Vector vals;

   int geom = pfes->GetParMesh()->GetElement(0)->GetGeometryType();
   const IntegrationPoint &cent = Geometries.GetCenter(geom);
   const FiniteElement *fe;
   ElementTransformation *eltransf;
   for (int i = 0; i < pfes->GetNE(); i++)
   {
      fe = pfes->GetFE(i);
      // Get the indices of dofs
      for (int j = 0; j < vdim; j ++)
      {
         vdofs[j] = i * vdim +j;
      }

      eltransf = pfes->GetElementTransformation(i);
      eltransf->SetIntPoint(&cent);
      vals.SetSize(vdofs.Size());
      coeff.Eval(vals, *eltransf, cent);

      if (fe->GetMapType() == 1 )
      {
        vals(i) *= eltransf->Weight();
      }
      GetTrueVector().SetSubVector(vdofs, vals);
   }
}

HypreParVector *ParCentGridFunction::GetTrueDofs() const
{
   HypreParVector *tv = dynamic_cast<ParGDSpace*>(pfes)->NewTrueDofVector();
   const SparseMatrix *R = fes->GetRestrictionMatrix();
   R->Mult(t_vec,*tv);
   return tv;
}

ParCentGridFunction &ParCentGridFunction::operator=(const Vector &v)
{
   MFEM_ASSERT(fes && v.Size() == fes->GetTrueVSize(), "");
   //Vector::operator=(v);
   ParGridFunction::operator=(v);
   return *this;
}

ParCentGridFunction &ParCentGridFunction::operator=(double value)
{
   ParGridFunction::operator=(value);
   return *this;
}
}