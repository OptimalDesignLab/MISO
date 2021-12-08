#include "RBFSpace.hpp"

using namespace std;
using namespace mfem;
using namespace mach;

namespace mfem
{

RBFSpace::RBFSpace(Mesh *m, const FiniteElementCollection *f, 
                   Array<Vector> center, int vdim,
                   int ordering, int degree)
   : SpaceType(m, f, vdim, ordering), basisCenter(center)
{
   // numBasis should not be greater than the number of elements
   numBasis = basisCenter.Size();
   polyOrder = degree;

   // check the basis centers
   for (int i = 0; i < numBasis; i++)
   {
      basisCenter[i].Print();
   }

   /// initialize the stencil/patch

}


} // end of namespace mfem