#include "evolver.hpp"

using namespace mfem;
using namespace std;

namespace mach
{

void ElementInv(const Vector &x, Vector &y)
{
   MFEM_ASSERT( x.Size() == y.Size(), "");
   for (int i = 0; i < x.Size(); ++i)
   {
      y(i) = 1.0/x(i);
   }
}

void HadamardProd(const Vector &v1, const Vector &v2, Vector &v)
{
   MFEM_ASSERT( v1.Size() == v2.Size() && v1.Size() == v.Size(), "");
   for (int i = 0; i < v.Size(); ++i)
   {
      v(i) = v1(i)*v2(i);
   }
}

LinearEvolver::LinearEvolver(MatrixType &m, MatrixType &k, ostream &outstream) //, const Vector &_b)
   : out(outstream), TimeDependentOperator(m.Height()), mass(m), stiff(k), z(m.Height()) //b(_b), z(_M.Size())
{
    // Here we extract the diagonal from the mass matrix and invert it
    //M.GetDiag(z);
    //cout << "minimum of z = " << z.Min() << endl;
    //cout << "maximum of z = " << z.Max() << endl;
    //ElementInv(z, Minv);
#ifdef MFEM_USE_MPI
   mass_prec.SetType(HypreSmoother::Jacobi);
   mass_solver.reset(new CGSolver(mass.GetComm()));
#else
   mass_solver.reset(new CGSolver());
#endif
   mass_solver->SetPreconditioner(mass_prec);
   mass_solver->SetOperator(mass);
   mass_solver->iterative_mode = false; // do not use second arg of Mult as guess
   mass_solver->SetRelTol(1e-9);
   mass_solver->SetAbsTol(0.0);
   mass_solver->SetMaxIter(100);
   mass_solver->SetPrintLevel(0);
}

void LinearEvolver::Mult(const Vector &x, Vector &y) const
{
   // y = M^{-1} (K x)
   //HadamardProd(Minv, x, y);
   stiff.Mult(x, z);
   mass_solver->Mult(z, y);
   //HadamardProd(Minv, z, y);
}

NonlinearEvolver::NonlinearEvolver(MatrixType &m, NonlinearFormType &r,
                                   double a)
   : TimeDependentOperator(m.Height()), mass(m), res(r), z(m.Height()), alpha(a)
{
#ifdef MFEM_USE_MPI
   mass_prec.SetType(HypreSmoother::Jacobi);
   mass_solver.reset(new CGSolver(mass.GetComm()));
#else
   mass_solver.reset(new CGSolver());
#endif
   mass_solver->SetPreconditioner(mass_prec);
   mass_solver->SetOperator(mass);
   mass_solver->iterative_mode = false; // do not use second arg of Mult as guess
   mass_solver->SetRelTol(1e-9);
   mass_solver->SetAbsTol(0.0);
   mass_solver->SetMaxIter(100);
   mass_solver->SetPrintLevel(0);
}

void NonlinearEvolver::Mult(const Vector &x, Vector &y) const
{
   res.Mult(x, z);
   mass_solver->Mult(z, y);
   y *= alpha;
}

}