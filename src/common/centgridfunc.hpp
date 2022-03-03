#ifndef MFEM_CENTGRIDFUNC
#define MFEM_CENTGRIDFUNC

#include "mfem.hpp"
#include "galer_diff.hpp"


namespace mfem
{
/// A derived grid function class used for store information on the element center
class CentGridFunction : public mfem::GridFunction
{
public:
   CentGridFunction() { }
   CentGridFunction(mfem::FiniteElementSpace *f);
   CentGridFunction(mfem::FiniteElementSpace *f, mfem::Vector center);

   virtual void ProjectCoefficient(mfem::VectorCoefficient &coeff);
   
   CentGridFunction &operator=(const Vector &v);
   CentGridFunction &operator=(double value);

private:
   int numBasis;
   Vector basisCenter;

};

} // end of namespace mfem

#endif