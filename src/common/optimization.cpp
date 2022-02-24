#include "optimization.hpp"
#include "default_options.hpp"

using namespace std;
using namespace mfem;


namespace mach
{

DGDOptimizer::DGDOptimizer(FiniteElementSpace *f,
									DGDSpace *f_dgd)
{
	inputSize = f_dgd->GetNDofs();
}



DGDOptimizer::~DGDOptimizer()
{
    cout << "Deleting the DGD optmization..." << '\n';
}


double DGDOptimizer::ComputeObject()
{
	Vector r(fes->GetVSize());
	SparseMatrix *prolong = fes_dgd->GetCP();
	prolong->Mult(*u_dgd,*u_full); 
	res_full->Mult(*u_full,r);
	double norm = r * r;
	return r;
}

Operator *DGDOperator::GetGradient()
{

}


}