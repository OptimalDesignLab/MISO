#ifndef MFEM_PARCENTGRIDFUNC
#define MFEM_PARCENTGRIDFUNC

#include "mfem.hpp"
namespace mfem
{

class ParCentGridFunction : public mfem::ParGridFunction
{
private:
	int proc; // temporal data created for print
public:
	ParCentGridFunction() { }
	ParCentGridFunction(mfem::ParFiniteElementSpace *pfes, int pr);
	virtual void ProjectCoefficient(mfem::VectorCoefficient &coeff);
	HypreParVector *GetTrueDofs() const;
	using GridFunction::GetTrueDofs;
	ParCentGridFunction &operator=(const Vector &v);
   ParCentGridFunction &operator=(double value);
};


} // end of namespace mfem

#endif