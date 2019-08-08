
#ifndef MFEM_INEXACTNEWTON
#define MFEM_INEXACTNEWTON

#include "mfem.hpp"
#include "solver.hpp"
using namespace mach;


namespace mfem
{
/// Inexact Newton's method solving F(x) = b with globalization.
/** The method GetGradient() must be implemented for the operator F.
    The preconditioner is used (in non-iterative mode) to evaluate
    the action of the inverse gradient of the operator. */
class InexactNewton : public mfem::NewtonSolver
{
protected:
   /// member vector saves the new x position.
   mfem::Vector x_new;
   /// Parameters for inexact newton method.
   double theta, eta, eta_max, t;
   const double theta_min = 0.1;
   const double theta_max = 0.5;
   
public:
   /// Constructor for Inexact Newton Solver.
   /// \param[in] eta0 - initial value of eta. Default is 1e-4
   /// \param[in] etam - maximum value of eta. Default is 0.9
   /// \param[in] t0 - initial value of t. Default is 1e-4
   /// \note the operator and inexact newton solver need to set in problem
   InexactNewton(double eta0 = 1e-4, double etam = 1e-1, double t0=1e-4)
   { eta = eta0; eta_max = etam; t = t0;}

   //Parallelization part currently is left unchange so far.
   // #ifdef MFEM_USE_MPI
   //    NewtonSolver(MPI_Comm _comm) : IterativeSolver(_comm) { }
   // #endif

   /// Set the operator and initialize x, r, c, x2.
   /// \param[in] op - reference to the problem operator
   virtual void SetOperator(const mfem::Operator &op);

   /// Set the linear solver for inverting the Jacobian.
   /// \param[in] solver - the inexact newton step solver
   /// \note This method is equivalent to calling SetPreconditioner().
   virtual void SetSolver(mfem::Solver &solver) { prec = &solver; }

   /// Solve the nonlinear system with right-hand side b.
   /// \param[in] b - the right-hand side vector.
   /// \param[in] x - starting point.
   virtual void Mult(const mfem::Vector &b, mfem::Vector &x);

   /// Back tracking globalization method making the step safer.
   /// \param[in] x - current x location.
   /// \param[in] b - the right-hand side vector.
   /// \param[in] norm - norm of the current residual.
   /// \returns the globalized stepsize.
   double ComputeStepSize(const mfem::Vector &x, const mfem::Vector &b, 
                        const double norm);

	 /// Get the Inexact Newton Solver.
    /// returns the nonlinear solver.
    /// \note there might be a way to avoid this.
    virtual mfem::Solver * GetSolver(){return prec;}
};

}

#endif
