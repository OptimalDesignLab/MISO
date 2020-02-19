#ifndef MFEM_CENTGRIDFUNC
#define MFEM_CENTGRIDFUNC

#include "mfem.hpp"



namespace mfem
{
/// A derived grid function class used for store information on the element center
class CentGridFunction : public mfem::GridFunction
{
public:
   CentGridFunction(mfem::FiniteElementSpace *f);
   // {
   //    fes = f;
   //    fec = NULL;
   //    sequence = f->GetSequence();
   //    UseDevice(true);
   // }

   virtual void ProjectCoefficient(mfem::VectorCoefficient &coeff);

};

} // end of namespace mfem

#endif