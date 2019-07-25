//#include "catch.hpp"
#include "../src/inexact_newton.hpp"

using namespace mfem;
using namespace mach;
using namespace std;

class QuadraticFunction;
class i_Solver;

// Class that define the problem itself.
class QuadraticFunction : public Operator
{
   private:
      DenseMatrix Jac;
   public:
      QuadraticFunction(ins s);
      // This Mult calculate the value of quadratic function.
      virtual void Mult(const Vector &x, Vector &y) const;
      virtual Operator &GetGradient(const Vector &k) const; 
      virtual ~QuadraticFunction();
};

QuadraticFunction::QuadraticFunction(int s)
: Operator(s), Jac(s)
{ }

QuadraticFunction::~QuadraticFunction()
{  }

QuadraticFunction::Mult(const Vector &x, Vector &y) const
{
   y[0]= x[0]*x[0];
}

Operator &QuadraticFunction::GetGradient(const Vector &k) const
{
   Jac(0,0) = 2*k(0);
   return *Jac;
}

// Class 2
class i_Solver : public IterativeSolver
{
   public:
      virtual void Mult(const Vector x, Vector c) const;
};

i_Solver::Mult(const Vector x, Vector c) const
{
   c[0] = 1/(*oper)(0,0) * (x[0]*x[0]-b[0]);
   // This function returns c = (DF/Dx)^-1 (F(x)-b)
}


int main(int argc, char * argv[])
{
   Vector b(1), x(1);
   b(0)=0.0;
   x(0)=10.0;
   const double abs_tol = 1e-8;
   const double rel_tol = 1e-6;
   QuadraticFunction *quadfunc(1);
   i_Solver *J_solver;

   /* Using inexact newton method solve the problem
      F(x) = x^2 = 0 */
   InexactNewton inexact_test;
   inexact_test.iterative_mod=false;
   inexact_test.SetOperator(*quadfunc);
   inexact_test.SetSolver(*J_solver);
   inexact_test.SetPrintLevel(1);
   inexact_test.SetAbsTol(abs_tol);
   inexact_test.SetRelTol(rel_rol);
   inexact_test.SetMaxIter(10);

   // Solve the problem.
   inexact_test.Mult(b,x);
   
   return 0;
}
