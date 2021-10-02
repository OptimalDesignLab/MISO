#ifndef MFEM_PARCENTGRIDFUNC
#define MFEM_PARCENTGRIDFUNC

#include "mfem.hpp"
#include "sbp_fe.hpp"


namespace mfem
{

class ParCentGridFunction : public mfem::ParGridFunction
{
private:
	int proc; // temporal data created for print
public:
	ParCentGridFunction() { }
	ParCentGridFunction(mfem::ParFiniteElementSpace *pfes, int pr = 0);
	virtual void ProjectCoefficient(mfem::VectorCoefficient &coeff);
	// HypreParVector *GetTrueDofs() const;
	// using GridFunction::GetTrueDofs;
	ParCentGridFunction &operator=(const Vector &v);
   ParCentGridFunction &operator=(double value);
};


} // end of namespace mfem

#endif