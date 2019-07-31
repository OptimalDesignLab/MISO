#include "catch.hpp"
#include "inexact_newton.hpp"
#include "mfem.hpp"
// This line is for testing
//using namespace mach;
//using namespace std;
//using namespace mfem;
class QuadraticFunction;
class i_Solver;

// Class that define the problem itself.
class QuadraticFunction : public mfem::Operator
{
   private:
     mutable mfem::DenseMatrix Jac;
     mfem::Vector b;
   public:
      QuadraticFunction(int s, mfem::Vector k);
      // This Mult calculate the value of quadratic function.
      virtual void Mult(const mfem::Vector &x, mfem::Vector &y) const;
      virtual mfem::Operator &GetGradient(const mfem::Vector &k) const; 
      virtual ~QuadraticFunction();
};

QuadraticFunction::QuadraticFunction(int s, mfem::Vector k)
: Operator(s), Jac(s), b(s)
{
   b(0) = k(0);
}

QuadraticFunction::~QuadraticFunction()
{
}

void QuadraticFunction::Mult(const mfem::Vector &x, mfem::Vector &y) const
{
   y[0]= x[0]*x[0]-b[0];
}

mfem::Operator &QuadraticFunction::GetGradient(const mfem::Vector &k) const
{
   Jac(0,0) = 2 * k(0);
   return static_cast<mfem::Operator&>(Jac);
}

// Class 2
class i_Solver : public mfem::IterativeSolver
{
   public:
      i_Solver() { };
      virtual void Mult(const mfem::Vector &x, mfem::Vector &c) const;
      virtual const mfem::Operator * GetOper(){return oper;}
};

void i_Solver::Mult(const mfem::Vector &r, mfem::Vector &c) const
{
	mfem::Vector temp(1),temp1(1);
   temp(0)=1.0;
   oper->Mult(temp, temp1);
   //c(0)=temp1(0);
   c(0) = 1.0/temp1(0) * r(0);
}

const double abs_tol=1e-8;
const double rel_tol=1e-10;
TEST_CASE(" Use Inexact Newton Method solving a 1D problem...","[inexact-newton]")
{

   // Solve the problem x^2 - b = 0;
   // Initial condition.
	mfem::Vector b(1), x(1),c(1),r(1);
	b(0)=1.0;
	x(0)=3.0;


   // declare the inexact newton solver.
   mfem::InexactNewton inexact_test;
   inexact_test.iterative_mode=false;
   inexact_test.SetPrintLevel(1);
   inexact_test.SetAbsTol(abs_tol);
   inexact_test.SetRelTol(rel_tol);
   inexact_test.SetMaxIter(5);

   // Operator constunction
   QuadraticFunction *quadfunc;
   quadfunc =new QuadraticFunction(1,b);

   // Solver construcrtion
   i_Solver *J_solve;
   J_solve = new i_Solver();
   J_solve->SetAbsTol(abs_tol);
   J_solve->SetRelTol(rel_tol);




   SECTION( "Set Operation, size check, and check the initial function value. ")
   {
      inexact_test.SetOperator(*quadfunc);
      quadfunc->Mult(x,r);
      REQUIRE(r(0)== x(0)*x(0)-b(0));
   }
   // Section tests passed, assign the operator.
   inexact_test.SetOperator(*quadfunc);
   r(0) = x(0) * x(0) - b(0);

   SECTION( "Set Solver, and check the initial derivative.  ")
	{
      mfem::Vector temp(1), temp1(1);
      temp(0)=1.0;
      J_solve->SetOperator(quadfunc->GetGradient(x));
      J_solve->GetOper()->Mult(temp,temp1);
      //J_solve->Mult(x,c);
      inexact_test.SetSolver(*J_solve);
      inexact_test.GetSolver()->Mult(r,c);
      REQUIRE(temp1(0) == 6.0);
      REQUIRE(c(0) == ((x(0)*x(0)-b(0))/2.0/x(0)));
      REQUIRE(c(0) == 8.0/6.0);
   }
   // Section tests passed, assign the solver.
   inexact_test.SetSolver(*J_solve);


   SECTION( " Solve simply equation " )
   {
      inexact_test.Mult(b,x);
      REQUIRE(x(0)-1 < 1e-3);
   }
	delete quadfunc;
   delete J_solve;
}
