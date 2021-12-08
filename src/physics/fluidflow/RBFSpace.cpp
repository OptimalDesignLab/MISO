#include "RBFSpace.hpp"

using namespace std;
using namespace mfem;
using namespace mach;

namespace mfem
{

RBFSpace::RBFSpace(Mesh *m, const FiniteElementCollection *f, int vdim,
                   int ordering, int degree)
   : SpaceType(m, f, vdim, ordering)
{
   numBasis = m->GetNE();
   polyOrder = degree;
}

} // end of namespace mfem