#include "evolver.hpp"
#include "utils.hpp"

using namespace mfem;
using namespace std;

namespace mach
{

LinearEvolver::LinearEvolver(MatrixType &m, MatrixType &k, ostream &outstream)
   : out(outstream), TimeDependentOperator(m.Height()), mass(m), stiff(k), z(m.Height())
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

ImplicitLinearEvolver::ImplicitLinearEvolver(MatrixType &m,
                                            MatrixType &k, 
                                            std::ostream &outstream)
   : out(outstream),  TimeDependentOperator(m.Height()), mass(m), stiff(k), z(m.Height())
{
#ifdef MFEM_USE_MPI
   mass_prec.SetType(HypreSmoother::Jacobi);
   mass_solver.reset(new CGSolver(mass.GetComm()));
#else
   mass_solver.reset(new CGSolver());
#endif
   // set parameters for the linear solver
   mass_solver->SetTol(1e-10);
   mass_solver->SetPrintLevel(-1);
   mass_solver->SetMaxIter(100);

}