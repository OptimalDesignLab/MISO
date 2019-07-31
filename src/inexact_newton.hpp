
#ifndef MFEM_INEXACTNEWTON
#define MFEM_INEXACTNEWTON

#include "mfem.hpp"
#include "solver.hpp"
using namespace mach;
namespace mfem
{
/// Newton's method for solving F(x)=b for a given operator F.
/** The method GetGradient() must be implemented for the operator F.
    The preconditioner is used (in non-iterative mode) to evaluate
    the action of the inverse gradient of the operator. */
class InexactNewton : public mfem::NewtonSolver
{
protected:
   // This operator is unnessary.
   const Operator* jac;
   mfem::Vector r2, x2;
   /* Caustion: Here we make norm a member function so that we
      can avoid evaulate the current norm in each iteration.*/
   double norm;
   double theta, eta;
   double eta_max;
   const double theta_min = 0.1;
   const double theta_max = 0.5;
   const double t = 1e-4;

public:
   InexactNewton(double eta0 = 0.01, double etam = 0.9)
    { eta = eta0; eta_max = etam; theta = 1e-4; }
   
// Parallelization part currently is left unchange so far.
// #ifdef MFEM_USE_MPI
//    NewtonSolver(MPI_Comm _comm) : IterativeSolver(_comm) { }
// #endif

   virtual void SetOperator(const mfem::Operator &op);
   /// Set the linear solver for inverting the Jacobian.
   /** This method is equivalent to calling SetPreconditioner(). */
   virtual void SetSolver(mfem::Solver &solver) { prec = &solver; }

   /// Solve the nonlinear system with right-hand side @a b.
   /** If `b.Size() != Height()`, then @a b is assumed to be zero. */
   /// This comment was added on July 9th.
   virtual void Mult(const mfem::Vector &b, mfem::Vector &x);

   /** @brief This method is overloaded in this derived classes to implement 
       other line search algorithms. Currently the line searching method is backtraching
       method with quadratic interpolation. */
   virtual double ComputeScalingFactor(const mfem::Vector &x, const mfem::Vector &b);

   /* This function set the jacbian operator. */
   virtual void SetJacobian(const mfem::Operator &op){jac = &op;}
	 
	 // Below here are functions that used for 
	 virtual const mfem::Operator * GetOper(){return oper;}
    virtual mfem::Solver * GetSolver(){return prec;}
};

}

#endif
