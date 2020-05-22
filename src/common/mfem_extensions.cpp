#include <iostream>

#include "mfem_extensions.hpp"
#include "evolver.hpp"
#include "utils.hpp"

using namespace mfem;
using namespace std;
using namespace mach;

namespace mach
{

void RRKImplicitMidpointSolver::Init(TimeDependentOperator &_f)
{
   ODESolver::Init(_f);
   k.SetSize(f->Width(), mem_type);
}

void RRKImplicitMidpointSolver::Step(Vector &x, double &t, double &dt)
{
   f->SetTime(t + dt/2);
   f->ImplicitSolve(dt/2, x, k);
   //f->ImplicitSolve(dt, x, k);
   //cout << "equation solved using regular midpont solver\n";
   // Set-up and solve the scalar nonlinear problem for the relaxation gamma
   EntropyConstrainedOperator *f_ode =
       dynamic_cast<EntropyConstrainedOperator *>(f);
   //cout << "x size is " << x.Size() << '\n';
   //cout << "x is empty? == " << x.GetMemory().Empty() << '\n';
   double delta_entropy = f_ode->EntropyChange(dt/2, x, k);
   //double delta_entropy = f_ode->EntropyChange(dt, x, k);
   //cout << "delta_entropy is " << delta_entropy << '\n';
   double entropy_old = f_ode->Entropy(x);
   //cout << "old entropy is " << entropy_old << '\n';
   mfem::Vector x_new(x.Size());
   //cout << "x_new size is " << x_new.Size() << '\n';
   auto entropyFun = [&](double gamma)
   {
      cout << "In lambda function: " << std::setprecision(14); 
      add(x, gamma*dt, k, x_new);
      double entropy = f_ode->Entropy(x_new);
      cout << "gamma = " << gamma << ": ";
      cout << "residual = " << entropy - entropy_old + gamma*dt*delta_entropy << endl;
      //cout << "new entropy is " << entropy << '\n';
      return entropy - entropy_old + gamma*dt*delta_entropy;
   };
   // TODO: tolerances and maxiter should be provided in some other way
   const double ftol = 1e-12;
   const double xtol = 1e-12;
   const int maxiter = 30;
   //double gamma = bisection(entropyFun, 0.50, 1.5, ftol, xtol, maxiter);
   double gamma = secant(entropyFun, 0.99, 1.01, ftol, xtol, maxiter);
   cout << "\tgamma = " << gamma << endl;
   x.Add(gamma*dt, k);
   t += gamma*dt;
}

} // namespace mach