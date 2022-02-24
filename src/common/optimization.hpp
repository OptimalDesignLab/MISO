#ifndef MACH_OPTIMIZATION
#define MACH_OPTIMIZATION

#include "solver.hpp"
#include "galer_diff.hpp"
namespace mach
{

class DGDOptimization : public mfem::NonlinearForm
{
public:
    /// class constructor
    DGDOptimization(mfem::FiniteElementSpace *fes,
	 					  mfem::DGDSpace *fes_dgd);
    
    /// evalute the objective function
    virtual void Mult(const mfem::Vector &x, mfem::Vector &y) const;
    /// get gradient function
    virtual Operator &GetGradient(const mfem::Vector &x) const;
    /// class destructor
   ~DGDOptimization();
protected:    
   /// do I want to keep all the objectives here,
   ///  or simply use the EulerSolver?

	int inputSize;
   
   mfem::DGDSpace* fes_dgd;
	// the constraints
	std::unique_ptr<NonlinearFormType> res;


};

} // end of namesapce
#endif