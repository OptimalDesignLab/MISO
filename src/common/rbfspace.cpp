#include "rbfspace.hpp"

using namespace std;
using namespace mfem;

namespace mfem
{

RBFSpace::RBFSpace(mesh *m, const FiniteElementCollection *f, int nb
                   int vdim, int ordering)
   : SpaceType(m, f, vdim, ordering)
{
   num_basis = nb;
}
} // end of namespace mfem