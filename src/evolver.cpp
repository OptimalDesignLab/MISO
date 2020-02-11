#include "evolver.hpp"
#include "utils.hpp"

#include <fstream>

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

ImplicitLinearEvolver::ImplicitLinearEvolver(const std::string &opt_file_name,
                                             MatrixType &m,
                                             MatrixType &k, 
                                             std::unique_ptr<LinearForm> b,
                                             std::ostream &outstream)
   : out(outstream),  TimeDependentOperator(m.Height()), mass(m), stiff(k), 
                                             force(move(b)), z(m.Height())
{
   // t = m + dt*k

   // get options
   nlohmann::json file_options;
   ifstream opts(opt_file_name);
   opts >> file_options;
   options.merge_patch(file_options);

	std::cout << "Setting Up Linear Solver..." << std::endl;
   t_prec.reset(new HypreSmoother());
#ifdef MFEM_USE_MPI
   t_prec->SetType(HypreSmoother::Jacobi);
   t_solver.reset(new CGSolver(stiff.GetComm()));
#else
   t_solver.reset(new CGSolver());
#endif
   // set parameters for the linear solver


   t_solver->iterative_mode = false;
   t_solver->SetRelTol(1e-8);//options["lin-solver"]["rel-tol"].get<double>());
   t_solver->SetAbsTol(0.0);//options["lin-solver"]["abs-tol"].get<double>());
   t_solver->SetMaxIter(500);//options["lin-solver"]["max-iter"].get<int>());
   t_solver->SetPrintLevel(1);//options["lin-solver"]["print-lvl"].get<int>());
   t_solver->SetPreconditioner(*t_prec);
}

void ImplicitLinearEvolver::ImplicitSolve(const double dt, const Vector &x, Vector &k)
{
   // if (T == NULL)
   // {
      T = Add(1.0, mass, dt, stiff);
      t_solver->SetOperator(*T);
   //}
   stiff.Mult(x, z);
   z.Neg();  
   z.Add(-1, *rhs);
   t_solver->Mult(z, k); 
   T = NULL;
}

}//namespace mach