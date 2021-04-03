#include <iostream>

#include "mfem_extensions.hpp"
#include "evolver.hpp"
#include "utils.hpp"

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
}