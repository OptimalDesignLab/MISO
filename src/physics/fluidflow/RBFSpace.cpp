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
                   Array<Vector*> center, double shape, int vdim, int e,
                   int ordering, int degree)
   : SpaceType(m, f, vdim, ordering), basisCenter(center),
     extra_basis(e)
{
   // numBasis should not be greater than the number of elements
   dim = m->Dimension();
   numBasis = center.Size();
   polyOrder = degree;

   switch(dim)
   {
      case 1: req_basis = polyOrder + 1; break;
      case 2: req_basis = (polyOrder+1) * (polyOrder+2) / 2; break;
      case 3: req_basis = (polyOrder+1)*(polyOrder+2)*(polyOrder+3) / 6; break;
      default: throw MachException("dim must be 1, 2 or 3.\n");
   }
   // initialize the stencil/patch
   req_basis += extra_basis;
   cout << "req_basis is " << req_basis << '\n';
   InitializeStencil();

   // initialize the shape parameter matrix
   cout << "Print the shapeParam.\n";
   shapeParam.SetSize(dim);
   for (int i = 0; i < dim; i++)
   {
      shapeParam(i,i) = shape;
   }
   shapeParam.Print(cout,dim);

   // initialize the prolongation matrix
   cP = new mfem::SparseMatrix(GetVSize(),vdim*numBasis);

   buildProlongationMatrix();
}

void RBFSpace::InitializeStencil()
{
   // initialize the all element centers for later used
   elementCenter.SetSize(GetMesh()->GetNE());
   elementBasisDist.SetSize(GetMesh()->GetNE());
   selectedBasis.SetSize(GetMesh()->GetNE());
   coef.SetSize(GetMesh()->GetNE());
   selectedElement.SetSize(numBasis);
   Vector diff;
   double dist;
   for (int i = 0; i < numBasis; i++)
   {
      selectedElement[i] = new Array<int>;
   }
   for (int i = 0; i < GetMesh()->GetNE(); i++)
   {
      elementCenter[i] = new Vector(dim);
      elementBasisDist[i] = new std::vector<double>;
      selectedBasis[i] = new Array<int>;
      GetMesh()->GetElementCenter(i,*elementCenter[i]);
      for (int j = 0; j < numBasis; j++)
      {
         diff = *basisCenter[j];
         diff -= *elementCenter[i];
         dist = diff.Norml2();
         elementBasisDist[i]->push_back(dist);
      }

   }

   // build element/basis stencil
   vector<size_t> temp;
   for (int i = 0; i < GetMesh()->GetNE(); i++)
   {
      temp = sort_indexes(*elementBasisDist[i]);
      for (int j = 0; j < req_basis; j++)
      {
         selectedBasis[i]->Append(temp[j]);
         selectedElement[temp[j]]->Append(i);
      }  
   }

   cout << "------Check the stencil------\n";
   cout << "------Basis center loca------\n";
   for (int i = 0; i < numBasis; i++)
   {  
      cout << "basis " << i << ": ";
      basisCenter[i]->Print();
   }
   cout << '\n';
   cout << "------Elem's  stencil------\n";
   for (int i = 0; i < GetMesh()->GetNE(); i++)
   {
      cout << "Element " << i << ": ";
      for (int j = 0; j < selectedBasis[i]->Size(); j++)
      {
         cout << (*selectedBasis[i])[j] << ' ';
      }
      cout << '\n';
   }
   cout << '\n';
   cout << "------Basis's  element------\n";
   for (int k = 0; k < numBasis; k++)
   {
      cout << "basis " << k << ": ";
      for (int l = 0; l < selectedElement[k]->Size(); l++)
      {
         cout << (*selectedElement[k])[l] << ' ';
      }
      cout << '\n';
   }
}

void RBFSpace::buildProlongationMatrix()
{
   // get the current element number of dofs
   // assume uniform type of element
   Array<Vector *> dofs_coord;
   const Element *el = mesh->GetElement(0);
   const FiniteElement *fe = fec->FiniteElementForGeometry(el->GetGeometryType());
   const int num_dofs = fe->GetDof();
   dofs_coord.SetSize(num_dofs);
   for (int k = 0; k <num_dofs; k++)
   {
      dofs_coord[k] = new Vector(dim);
   }

   // loop over element to build local and global prolongation matrix
   for (int i = 0; i < 1; i++)
   {
      cout << "element " << i << ": \n";
      // 1. Get the quad and basis centers
      buildDofMat(i, num_dofs, fe, dofs_coord);
      cout << "dof points:\n";
      for (int j = 0; j < num_dofs; j++)
      {
         dofs_coord[j]->Print();
      }
      cout << endl;

      // 2. build the interpolation matrix
      solveProlongationCoefficient(i,num_dofs,dofs_coord);

      // // 3. Assemble prolongation matrix
      // AssembleProlongationMatrix();
   }

   // free the aux variable
   for (int k = 0; k < num_dofs; k++)
   {
      delete dofs_coord[k];
   }
}


void RBFSpace::buildDofMat(int el_id, const int num_dofs,
                           const FiniteElement *fe,
                           Array<Vector *> &dofs_coord) const
{
   Vector coord(dim);
   ElementTransformation *eltransf = mesh->GetElementTransformation(el_id);
   for (int i = 0; i < num_dofs; i++)
   {
      eltransf->Transform(fe->GetNodes().IntPoint(i), coord);
      *dofs_coord[i] = coord;
   }
}


void RBFSpace::solveProlongationCoefficient(const int el_id, const int numDofs,
                                            const Array<Vector *> &dofs_coord)
{
   // some basic inf
   int numLocalBasis = selectedBasis[el_id]->Size();
   int numPolyBasis=-1;
   switch(dim)
   {
      case 1: numPolyBasis = polyOrder + 1; break;
      case 2: numPolyBasis = (polyOrder+1) * (polyOrder+2) / 2; break;
      case 3: numPolyBasis = (polyOrder+1)*(polyOrder+2)*(polyOrder+3)/6; break;
      default: throw MachException("dim must be 1, 2 or 3.\n");
   }
   
   // declare the basis matrix
   DenseMatrix W(numLocalBasis,numLocalBasis);
   DenseMatrix V(numLocalBasis,numPolyBasis);
   DenseMatrix Wn(numDofs, numLocalBasis);
   DenseMatrix Vn(numDofs, numPolyBasis);
   DenseMatrix WV(numLocalBasis+numPolyBasis, numLocalBasis+numPolyBasis);
   DenseMatrix WnVn(numDofs,numLocalBasis+numPolyBasis);
   coef[el_id] = new DenseMatrix(numLocalBasis+numPolyBasis,numLocalBasis);
   for (int i = 0; i < numLocalBasis; i++)
   {
      (*coef[el_id])(i,i) = 1.0;
   }
   cout << "coefficient is:\n";
   coef[el_id]->Print();

   // RBF matrix section
   buildElementRadialBasisMat(el_id,numDofs,dofs_coord,W,Wn);
   buildElementPolyBasisMat(el_id,polyOrder,numDofs,dofs_coord,V,Vn);
   buildWVMat(W,V,WV);
   buildWnVnMat(Wn,Vn,WnVn);
   cout << "WV mat:\n";
   WV.Print(cout,WV.Width());
   cout << "WVn mat:\n";
   WnVn.Print(cout,WnVn.Width());


   // Solve the coefficient
   buildRBFInterpolation(numLocalBasis,numPolyBasis,WV,*coef[el_id]);
   cout << "coefficient is:\n";
   coef[el_id]->Print();

   // Get the local prolongation operator ?
}

void RBFSpace::buildElementRadialBasisMat(const int el_id,
                                          const int numDofs,
                                          const Array<Vector *> &dofs_coord,
                                          DenseMatrix &W, DenseMatrix &Wn)
{
   const int numLocalBasis = selectedBasis[el_id]->Size();
   int i,j,center_id, loc_id;
   Vector center_coord, loc_coord;

   // loop over the column of basis mat
   for (i = 0; i < numLocalBasis; i++)
   {
      center_id = (*selectedBasis[el_id])[i];
      center_coord = *basisCenter[center_id];

      // coefficient matrix
      // loop over the row
      for (j = 0; j < numLocalBasis; j++)
      {
         loc_id = (*selectedBasis[el_id])[j];
         loc_coord = *basisCenter[loc_id];
         W(j,i) = radialBasisKernel(loc_coord,shapeParam,center_coord);
      }

      // actual prolongation matrix
      for (j = 0; j < numDofs; j++)
      {
         Wn(j,i) = radialBasisKernel(*dofs_coord[j],shapeParam,center_coord);
      }
   }
}

void RBFSpace::buildElementPolyBasisMat(const int el_id, const int numPolyBasis,
                                        const int numDofs,
                                        const Array<Vector *> &dofs_coord,
                                        DenseMatrix &V, DenseMatrix &Vn)
{
   const int numLocalBasis = selectedBasis[el_id]->Size();

   int i,j,k,l;
   double dx,dy,dz;
   int loc_id;
   Vector loc_coord;
   Vector el_center = *elementCenter[el_id];

   if (1 == dim)
   {
      // form the V matrix
      for (i = 0; i < numLocalBasis; i++)
      {
         loc_id = (*selectedBasis[el_id])[i];
         loc_coord = *basisCenter[loc_id];
         dx = loc_coord[0] - el_center[0];
         for (j = 0; j <= polyOrder; j++)
         {
            V(i,j) = pow(dx,j);
         }
      }

      // form the Vn matrix
      for (i = 0; i < numDofs; i++)
      {
         loc_coord = *dofs_coord[i];
         dx = loc_coord[0] - el_center[0];
         for (j = 0; j <= polyOrder; j++)
         {
            Vn(i,j) = pow(dx,j);
         }
      }
   }
   else if (2 == dim)
   {
      // form the V matrix
      for (i = 0; i < numLocalBasis; i++)
      {
         loc_id = (*selectedBasis[el_id])[i];
         loc_coord = *basisCenter[loc_id];
         dx = loc_coord[0] - el_center[0];
         dy = loc_coord[1] - el_center[1];
         int col = 0;
         for (j = 0; j <= polyOrder; j++)
         {
            for (k = 0; k <= j; k++)
            {
               V(i,col) = pow(dx,j-k)*pow(dy,k);
               col++;
            }
         }
      }

      // form the Vn matrix
      for (i = 0; i < numDofs; i++)
      {
         loc_coord = *dofs_coord[i];
         dx = loc_coord[0] - el_center[0];
         dy = loc_coord[1] - el_center[1];
         int col = 0;
         for (j = 0; j <= polyOrder; j++)
         {
            for (k = 0; k <= j; k++)
            {
               Vn(i,col) = pow(dx,j-k)*pow(dy,k);
               col++;
            }
         }
      }

   }
   else if (3 == dim)
   {
      // form the V matrix
      for (i = 0; i < numLocalBasis; i++)
      {
         loc_id = (*selectedBasis[el_id])[i];
         loc_coord = *basisCenter[loc_id];
         dx = loc_coord[0] - el_center[0];
         dy = loc_coord[1] - el_center[1];
         dz = loc_coord[2] - el_center[2];
         int col = 0;
         for (j = 0; j <= polyOrder; j++)
         {
            for (k = 0; k <= j; k++)
            {
               for (l = 0; l <= j-k; l++)
               {
                  V(i,col) = pow(dx,j-k-l)*pow(dy,l)*pow(dz,k);
                  col++;
               }
            }
         }
      }


      // form the Vn matrix
      for (i = 0; i < numDofs; i++)
      {
         loc_coord = *dofs_coord[i];
         dx = loc_coord[0] - el_center[0];
         dy = loc_coord[1] - el_center[1];
         dz = loc_coord[2] - el_center[2];
         int col = 0;
         for (j = 0; j <= polyOrder; j++)
         {
            for (k = 0; k <= j; k++)
            {
               for (l = 0; l <= j-k; l++)
               {
                  Vn(i,col) = pow(dx,j-k-l)*pow(dy,l)*pow(dz,k);
                  col++;
               }
            }
         }
      }
   }
}

void RBFSpace::buildWVMat(const DenseMatrix &W, const DenseMatrix &V,
                          DenseMatrix &WV)
{
   int i,j;
   int numLocalBasis = W.Width();
   int numPolyBasis =  V.Width();

   // fill W
   for (i = 0; i < numLocalBasis; i++)
   {
      for (j = 0; j < numLocalBasis; j++)
      {
         WV(j,i) = W(j,i);
      }
   }

   // fill V and V'
   for (i = 0; i < numPolyBasis; i ++)
   {
      for (j = 0; j < numLocalBasis; j++)
      {
         // V
         WV(j,i+numLocalBasis) = V(j,i);
         
         // V'
         WV(i+numLocalBasis,j) = V(j,i);
      }
   }
}


void RBFSpace::buildWnVnMat(const DenseMatrix &Wn, const DenseMatrix &Vn,
                  DenseMatrix &WnVn)
{
   int i, j;
   int numDofs = Wn.Height();
   int numLocalBasis = Wn.Width();
   int numPolyBasis = Vn.Width();

   // fill Wn
   for (i = 0; i < numLocalBasis; i++)
   {
      for (j = 0; j < numDofs; j++)
      {
         WnVn(j,i) = Wn(j,i);
      }
   }

   // fill Vn
   for (i = 0; i < numPolyBasis; i++)
   {
      for (j = 0; j < numDofs; j++)
      {
         WnVn(j,i+numLocalBasis) = Vn(j,i);
      }
   }
}
RBFSpace::~RBFSpace()
{
   for (int k = 0; k < GetMesh()->GetNE(); k++)
   {
      delete selectedBasis[k];
      delete elementCenter[k];
      delete elementBasisDist[k];
      delete coef[k];
   }

   for (int k = 0; k < numBasis; k++)
   {
      delete selectedElement[k];
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