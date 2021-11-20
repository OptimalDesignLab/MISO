#ifndef MFEM_PCENTGRIDFUNC
#define MFEM_PCENTGRIDFUNC

#include "mfem.hpp"
using namespace mfem;
namespace mfem
{
/// A derived grid function class used for store information on the element
/// center
class ParCentGridFunction : public mfem::ParGridFunction
{
public:
   ParCentGridFunction(mfem::ParFiniteElementSpace *f);
   virtual void ProjectCoefficient(mfem::VectorCoefficient &coeff);
   // using mfem::GridFunction::GetTrueDofs;
   using mfem::ParGridFunction::SetFromTrueDofs;
   /// Returns the true dofs in a new HypreParVector
   virtual HypreParVector *GetTrueDofs() const;
   ParCentGridFunction &operator=(const Vector &v);
   ParCentGridFunction &operator=(double value);
   void GetTrueDofs(Vector &tv) const
   {
      // const SparseMatrix *R = fes->GetRestrictionMatrix();
      // tv.SetSize(R->Width());
      // R->Mult(*this, tv);
      tv.MakeRef(const_cast<ParCentGridFunction &>(*this), 0, size);
   }

protected:
   mfem::ParFiniteElementSpace *fes;
};
}  // namespace mfem
#endif