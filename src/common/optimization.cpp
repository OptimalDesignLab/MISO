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


double DGDOptimizer::GetEnergy(const Vector &x) const
{
	Vector r(fes->GetVSize());
	SparseMatrix *prolong = fes_dgd->GetCP();
	prolong->Mult(*u_dgd,*u_full); 
	res_full->Mult(*u_full,r);
	return r * r;
}
              
Operator *DGDOperator::GetGradient(const Vector &x) const;
{
	// dJ/dc = pJ/pc - pJ/puc * (pR_dgd/puc)^{-1} * pR_dgd/pc
	Vector pJpc(numVar);
	Vector pJpuc(numBasis);
	
	/// first compute some variables that used multiple times
	// 1. get pRpu
	SparseMatrix *pRpu = res_full->GetGradient(*uc);

	// 2. compute full residual
	Vector r(fes->GetVSize());
	res_full->Mult(*u_full,r);

	/// loop over all design variables
	Vector ppupc_col(fes->GetVSize());
	Vector dptpct_col(numBasis);
	SparseMatrix dPdci(fes->GetVSize(),numBasis);
	DenseMatrix pPupc(fes->GetVSize(),numVar);
	DenseMatrix pPtpcR(numBasis,numVar);
	for (int i = 0; i < numVar; i++)
	{
		// get dpdc
		fes_dgd->GetdPdc(i,dPdci);

		dPdci.Mult(*uc,ppupc_col);
		pPupc.SetCol(i,ppupc_col);

		dPdci.MultTranspose(r,dptpcr);
		pPtpcR.SetCol(i,dptpcr_col);

		// clean data in dPdc
		dPdci.Destroy();
	}



	// compute pJ/pc
	Vector temp_vec1(fes->GetVSize());
	pRpu->MultTranspose(r,temp_vec1);
	pPupc.MultTranspose(temp_vec1,pJpc);
	pJpc *= 2.0;

	// compute pJ/puc
	SparseMatrix *P = fes_dgd->GetCP();
	P->MultTranspose(temp_vec1,pJpuc);

	// compute pR_dgd / puc
	DenseMatrix *P_dense = P->ToDense();
	DenseMatrix *pR_dgdpuc = RAP(*pRpu,*P_dense);

	// compute pR_dgd / pc
	DenseMatrix *pR_dgdpc = Mult(*pRpu,pPupc);
}


}