#ifndef MFEM_RBFGRIDFUNC
#define MFEM_RBFGRIDFUNC

#include "mfem.hpp"
#include <functional>


namespace mfem
{
/// A derived grid function class used for store information on the element center
class RBFGridFunction : public mfem::GridFunction
{
public:
   RBFGridFunction() { }
   RBFGridFunction(mfem::FiniteElementSpace *f, mfem::Array<mfem::Vector *> &center,
                    std::function<void(const mfem::Vector &, mfem::Vector &)> F);

   virtual void ProjectCoefficient();
   
   RBFGridFunction &operator=(const Vector &v);
   RBFGridFunction &operator=(double value);
private:
   mfem::Array<mfem::Vector *> basisCenter;
   std::function<void(const mfem::Vector &, mfem::Vector &)> Function;
};

} // end of namespace mfem

#endif