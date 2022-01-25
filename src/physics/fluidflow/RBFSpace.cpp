#include "RBFSpace.hpp"
#include "utils.hpp"
#include <numeric> 
#include <algorithm> 

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
      elementBasisDist[i] = new std::vector<double>;
      GetMesh()->GetElementCenter(i,*elementCenter[i]);
      for (int j = 0; j < numBasis; j++)
      {
         diff = *basisCenter[j];
         diff -= *elementCenter[i];
         dist = diff.Norml2();
         elementBasisDist[i]->push_back(dist);
      }

   }


   cout << "Check the initial stencil\n";
   // check element <---> basis center distance
   vector<size_t> temp;
   for (int i = 0; i < GetMesh()->GetNE(); i++)
   {
      cout << "element " << i << ": ";
      for (int j = 0; j < numBasis; j++)
      {
         cout << (*elementBasisDist[i])[j] << ' ';
      }
      cout << "Then sort.\n";
      temp = sort_indexes(*elementBasisDist[i]);
      for (int j = 0; j < numBasis; j++)
      {
         cout << temp[j] << ' ';
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

vector<size_t> RBFSpace::sort_indexes(const vector<double> &v)
{

  // initialize original index locations
  vector<size_t> idx(v.size());
  iota(idx.begin(), idx.end(), 0);

  // sort indexes based on comparing values in v
  // using std::stable_sort instead of std::sort
  // to avoid unnecessary index re-orderings
  // when v contains elements of equal values 
  stable_sort(idx.begin(), idx.end(),
       [&v](size_t i1, size_t i2) {return v[i1] < v[i2];});

  return idx;
}

} // end of namespace mfem