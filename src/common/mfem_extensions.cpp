#include <iostream>

#include "mfem.hpp"

#include "evolver.hpp"
#include "utils.hpp"
#include "mfem_extensions.hpp"

using namespace mfem;
using namespace std;
using namespace mach;

namespace mach
{
void PseudoTransientSolver::Init(TimeDependentOperator &_f)
{
   ODESolver::Init(_f);
}

void PseudoTransientSolver::Step(Vector &x, double &t, double &dt)
{
   f->SetTime(t + dt);
   k.SetSize(x.Size(), mem_type);
   f->ImplicitSolve(dt, x, k);
   x.Add(dt, k);
   t += dt;
}

void RRKImplicitMidpointSolver::Init(TimeDependentOperator &_f)
{
   ODESolver::Init(_f);
}

void RRKImplicitMidpointSolver::Step(Vector &x, double &t, double &dt)
{
   auto *f_ode = dynamic_cast<EntropyConstrainedOperator *>(f);
   f_ode->SetTime(t + dt / 2);
   k.SetSize(x.Size(), mem_type);
   f_ode->ImplicitSolve(dt, dt / 2, x, k);

   // Set-up and solve the scalar nonlinear problem for the relaxation gamma
   // cout << "x size is " << x.Size() << '\n';
   // cout << "x is empty? == " << x.GetMemory().Empty() << '\n';
   double delta_entropy = f_ode->EntropyChange(dt / 2, x, k);
   // double delta_entropy = f_ode->EntropyChange(dt, x, k);
   *out << "delta_entropy is " << delta_entropy << '\n';
   double entropy_old = f_ode->Entropy(x);
   *out << "old entropy is " << entropy_old << '\n';
   mfem::Vector x_new(x.Size());
   // cout << "x_new size is " << x_new.Size() << '\n';
   auto entropyFun = [&](double gamma)
   {
      *out << "In lambda function: " << std::setprecision(14);
      add(x, gamma * dt, k, x_new);
      double entropy = f_ode->Entropy(x_new);
      *out << "gamma = " << gamma << ": ";
      *out << "residual = "
           << entropy - entropy_old + gamma * dt * delta_entropy << endl;
      // cout << "new entropy is " << entropy << '\n';
      return entropy - entropy_old + gamma * dt * delta_entropy;
   };
   // TODO: tolerances and maxiter should be provided in some other way
   const double ftol = 1e-12;
   const double xtol = 1e-12;
   const int maxiter = 30;
   // double gamma = bisection(entropyFun, 0.50, 1.5, ftol, xtol, maxiter);
   double gamma = secant(entropyFun, 0.99, 1.01, ftol, xtol, maxiter);
   *out << "\tgamma = " << gamma << endl;
   x.Add(gamma * dt, k);
   t += gamma * dt;
}

}  // namespace mach
