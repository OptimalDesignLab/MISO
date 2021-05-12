#ifndef MFEM_CENTGRIDFUNC
#define MFEM_CENTGRIDFUNC

#include "mfem.hpp"


namespace mfem
{
/// A derived grid function class used for store information on the element center
class CentGridFunction : public mfem::GridFunction
{
public:
   CentGridFunction() { }
   CentGridFunction(mfem::FiniteElementSpace *f);

   virtual void ProjectCoefficient(mfem::VectorCoefficient &coeff);
   
   CentGridFunction &operator=(const Vector &v);
   CentGridFunction &operator=(double value);
};

} // end of namespace mfem

#endif