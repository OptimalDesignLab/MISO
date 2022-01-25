#include "RBFSpace.hpp"
#include "utils.hpp"

using namespace std;
using namespace mfem;
using namespace mach;

namespace mfem
{

RBFSpace::RBFSpace(Mesh *m, const FiniteElementCollection *f, 
                   Array<Vector*> center, int vdim,
                   int ordering, int degree)
   : SpaceType(m, f, vdim, ordering), basisCenter(center)
{
   // numBasis should not be greater than the number of elements
   dim = m->Dimension();
   numBasis = center.Size();
   polyOrder = degree;

   switch(dim)
   {
      case 1: req_nel = polyOrder + 1; break;
      case 2: req_nel = (polyOrder+1) * (polyOrder+2) / 2; break;
      case 3: throw MachException("Not implemeneted yet.\n"); break;
      default: throw MachException("dim must be 1, 2 or 3.\n");
   }

   for (int i = 0; i < numBasis; i++)
   {
      basisCenter[i]->Print();
   }

   // initialize the stencil/patch
   InitializeStencil();

}

void RBFSpace::InitializeStencil()
{
   // initialize the all element centers for later used
   elementCenter.SetSize(GetMesh()->GetNE());
   elementBasisDist.SetSize(GetMesh()->GetNE());
   Vector diff;
   double dist;
   for (int i = 0; i < GetMesh()->GetNE(); i++)
   {
      elementCenter[i] = new Vector(dim);
      elementBasisDist[i] = new std::map<int, double>;
      GetMesh()->GetElementCenter(i,*elementCenter[i]);
      for (int j = 0; j < numBasis; j++)
      {
         diff = *basisCenter[j];
         diff -= *elementCenter[i];
         dist = diff.Norml2();
         elementBasisDist[i]->insert({j, dist});
      }
   }

   cout << "Check the initial stencil\n";
   // check element <---> basis center distance
   for (int i = 0; i < GetMesh()->GetNE(); i++)
   {
      cout << "element " << i << ": ";
      for (int j = 0; j < numBasis; j++)
      {
         cout << (*elementBasisDist[i])[j] << ' ';
      }
      cout << '\n';
   }


   
}

RBFSpace::~RBFSpace()
{
   for (int k = 0; k < GetMesh()->GetNE(); k++)
   {
      delete elementCenter[k];
      delete elementBasisDist[k];
   }

}

} // end of namespace mfem