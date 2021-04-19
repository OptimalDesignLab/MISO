#ifndef MFEM_PARCENTGRIDFUNC
#define MFEM_PARCENTGRIDFUNC

#include "mfem.hpp"
namespace mfem
{

class ParCentGridFunction : public mfem::ParGridFunction
{
public:
	ParCentGridFunction() { }
	ParCentGridFunction(mfem::ParFiniteElementSpace *pfes);
	virtual void ProjectCoefficient(mfem::VectorCoefficient &coeff);
	//HypreParVector *GetTrueDofs() const;

	ParCentGridFunction &operator=(const Vector &v);
   ParCentGridFunction &operator=(double value);
};


} // end of namespace mfem

#endif