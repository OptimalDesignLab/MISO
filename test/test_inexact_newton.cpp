/* This script tests the inexact newton method with solving
   a nonlinear problem:
            F(x) = -1/3 * x^3 + x^2 - x = -1
*/

#include "catch.hpp"
#include "inexact_newton.hpp"
#include "mfem.hpp"


class nonlinearFunc;
class nonlinearSolver;

// Class that define the problem itself.
class nonlinearFunc: public mfem::Operator
{
   private:
     mutable mfem::DenseMatrix Jac;
     //mfem::Vector b;
   public:
      nonlinearFunc(int s);
      // This Mult calculate the value of quadratic function.
      virtual void Mult(const mfem::Vector &x, mfem::Vector &y) const;
      virtual mfem::Operator &GetGradient(const mfem::Vector &k) const; 
      virtual ~nonlinearFunc();
};

nonlinearFunc::nonlinearFunc(int s)
: Operator(s), Jac(s)
{
}

nonlinearFunc::~nonlinearFunc()
{
}

void nonlinearFunc::Mult(const mfem::Vector &x, mfem::Vector &y) const
{
   y(0)= -1.0/3.0*x(0)*x(0)*x(0) + x(0)*x(0) - x(0);
   //y(0)= x(0) + 1.0/x(0);
}

mfem::Operator &nonlinearFunc::GetGradient(const mfem::Vector &k) const
{
   Jac(0,0) = -k(0)*k(0) + 2.0 * k(0) - 1.0;
   //Jac(0,0) = 1.0 - 1.0/(k(0)*k(0));
   return static_cast<mfem::Operator&>(Jac);
}

// Class 2
class nonlinearSolver : public mfem::IterativeSolver
{
   private:
      double iter_norm;
   public:
      nonlinearSolver() { };
      virtual void Mult(const mfem::Vector &x, mfem::Vector &c) const;
      virtual const mfem::Operator * GetOper(){return oper;}
};

void nonlinearSolver::Mult(const mfem::Vector &r, mfem::Vector &c) const
{
	mfem::Vector temp(1),temp1(1);
   temp(0)=1.0;
   oper->Mult(temp, temp1);
   c(0) = r(0)/temp1(0);

   // below are solve that satify ||F(u_k) + F'(u_k)s_k|| <= eta ||F(u_k)||
   // double norm = Norm(r);
   // double rhs = rel_tol * norm;
   // if ( r(0) >= 0)
   // {
   //    c(0) = -(rhs - r(0))/temp1(0);
   // }
   // else
   // {
   //    c(0) = -( - rhs - r(0))/temp1(0);
   // }
}

const double abs_tol=1e-12;
const double rel_tol=1e-10;
TEST_CASE(" Use Inexact Newton Method solving another 1D problem...",
         "[inexact-newton]")
{

   // Solve the problem x^2 - b = 0;
   // Initial condition.
	mfem::Vector b(1), x(1),c(1),r(1);
	b(0)=-1.0;
	x(0)=0.50;


   // declare the inexact newton solver.
   mfem::InexactNewton inexact_test(1e-4, 1e-1, 1e-4);
   inexact_test.iterative_mode=true;
   inexact_test.SetPrintLevel(-1);
   inexact_test.SetAbsTol(abs_tol);
   inexact_test.SetRelTol(rel_tol);
   inexact_test.SetMaxIter(100);

   // Operator constunction
   nonlinearFunc *quadfunc;
   quadfunc =new nonlinearFunc(1);

   // Solver construcrtion
   nonlinearSolver *J_solve;
   J_solve = new nonlinearSolver();
   J_solve->SetAbsTol(abs_tol);
   J_solve->SetRelTol(rel_tol);




   SECTION( "Set Operation, size check, and check the initial function value. ")
   {
      inexact_test.SetOperator(*quadfunc);
      quadfunc->Mult(x,r);
      REQUIRE(r(0)== -1.0/3.0*x(0)*x(0)*x(0) + x(0)*x(0)-x(0));
   }
   // Section tests passed, assign the operator.
   inexact_test.SetOperator(*quadfunc);
   r(0) = -1.0/3.0*x(0)*x(0)*x(0) + x(0)*x(0) - x(0) + 1;

   SECTION( "Set Solver, and check the initial derivative.  ")
	{
      mfem::Vector temp(1), temp1(1);
      temp(0)=1.0;
      double der = -x(0)*x(0) + 2.0*x(0) - 1.0;
      J_solve->SetOperator(quadfunc->GetGradient(x));
      J_solve->GetOper()->Mult(temp,temp1);

      inexact_test.SetSolver(*J_solve);
      inexact_test.GetSolver()->Mult(r,c);
      REQUIRE(temp1(0) == der);
      REQUIRE(c(0) == r(0)/der);
   }
   // Section tests passed, assign the solver.
   inexact_test.SetSolver(*J_solve);


   SECTION( " Solve simply equation " )
   {
      // Solve the 1d problem.
      inexact_test.Mult(b,x);
      REQUIRE(inexact_test.GetConverged()==1);
   }
	delete quadfunc;
   delete J_solve;
}
