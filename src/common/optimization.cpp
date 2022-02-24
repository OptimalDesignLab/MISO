#include "optimization.hpp"
#include "default_options.hpp"

using namespace std;
using namespace mfem;


namespace mach
{

DGDOptimization::DGDOptimization(FiniteElementSpace *f,
											DGDSpace *f_dgd)
	: NonlinearForm(f), fes_dgd(f_dgd)
{
	inputSize = fes_dgd->GetNDofs();
}



DGDOptimization::~DGDOptimization()
{
    cout << "Deleting the DGD optmization..." << '\n';
}


}