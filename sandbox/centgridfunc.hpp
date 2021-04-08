#include "mfem.hpp"

using namespace mfem;

/// A derived grid function class used for store information on the element center
class ParCentGridFunction : public mfem::ParGridFunction
{
public:
   ParCentGridFunction(mfem::ParFiniteElementSpace *f);
   virtual void ProjectCoefficient(mfem::VectorCoefficient &coeff);
   ParCentGridFunction &operator=(const Vector &v);
   ParCentGridFunction &operator=(double value);
};