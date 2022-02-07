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
      case 1: numPolyBasis = polyOrder + 1;  break;
      case 2: numPolyBasis = (polyOrder+1) * (polyOrder+2) / 2; break;
      case 3: numPolyBasis = (polyOrder+1)*(polyOrder+2)*(polyOrder+3) / 6; break;
      default: throw MachException("dim must be 1, 2 or 3.\n");
   }
   req_basis = numPolyBasis + extra_basis;
   cout << "Number of polynomial basis is " << numPolyBasis << '\n';
   cout << "Number of required basis is " << req_basis << '\n';

   // initialize the stencil/patch
   InitializeStencil();

   // initialize the shape parameter matrix
   shapeParam.SetSize(dim);
   for (int i = 0; i < dim; i++)
   {
      shapeParam(i,i) = shape;
   }

   // initialize the prolongation matrix
   cP = new mfem::SparseMatrix(GetVSize(),vdim*numBasis);
   buildRBFProlongation();
   cP->Finalize();
   cP_is_set = true;
   cout << "Check cP size: " << cP->Height() << " x " << cP->Width() << '\n';
   ofstream cp_save("RBF_P.txt");
   cP->PrintMatlab(cp_save);
   cp_save.close();

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
      coef[i] = new DenseMatrix(req_basis+numPolyBasis,req_basis);
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

   // cout << "------Check the stencil------\n";
   // cout << "------Basis center loca------\n";
   // for (int i = 0; i < numBasis; i++)
   // {  
   //    cout << "basis " << i << ": ";
   //    basisCenter[i]->Print();
   // }
   // cout << '\n';
   // cout << "------Elem's  stencil------\n";
   // for (int i = 0; i < GetMesh()->GetNE(); i++)
   // {
   //    cout << "Element " << i << ": ";
   //    for (int j = 0; j < selectedBasis[i]->Size(); j++)
   //    {
   //       cout << (*selectedBasis[i])[j] << ' ';
   //    }
   //    cout << '\n';
   // }
   // cout << '\n';
   // cout << "------Basis's  element------\n";
   // for (int k = 0; k < numBasis; k++)
   // {
   //    cout << "basis " << k << ": ";
   //    for (int l = 0; l < selectedElement[k]->Size(); l++)
   //    {
   //       cout << (*selectedElement[k])[l] << ' ';
   //    }
   //    cout << '\n';
   // }
}

void RBFSpace::buildRBFProlongation()
{
   // declare soma matrix variables
   DenseMatrix W,V;
   DenseMatrix Wn,Vn;
   DenseMatrix WV,WnVn;
   DenseMatrix localMat;
   // loop over element to build local and global prolongation matrix
   for (int i = 0; i < GetMesh()->GetNE(); i++)
   {
      // 1. build basis matrix
      buildDataMat(i,W,V,Wn,Vn,WV,WnVn);
      
      // 2. build the interpolation matrix
      solveLocalProlongationMat(i,WV,WnVn,localMat);


      // 3. Assemble prolongation matrix
      AssembleProlongationMatrix(i,localMat);
   }
}

void RBFSpace::buildDataMat(int el_id, DenseMatrix &W, DenseMatrix &V,
                            DenseMatrix &Wn, DenseMatrix &Vn,
                            DenseMatrix &WV, DenseMatrix &WnVn)
{
   // get element related data
   const Element *el = mesh->GetElement(el_id);
   const FiniteElement *fe = fec->FiniteElementForGeometry(el->GetGeometryType());
   const int numDofs = fe->GetDof();
   ElementTransformation *eltransf = mesh->GetElementTransformation(el_id);
   
   // get the dofs coord
   Array<Vector *> dofs_coord;
   dofs_coord.SetSize(numDofs);
   Vector coord(dim);
   for (int k = 0; k <numDofs; k++)
   {
      dofs_coord[k] = new Vector(dim);
      eltransf->Transform(fe->GetNodes().IntPoint(k), coord);
      *dofs_coord[k] = coord;
   }

   // build the data mat
   const int numLocalBasis = selectedBasis[el_id]->Size(); 

   W.SetSize(numLocalBasis,numLocalBasis);
   V.SetSize(numLocalBasis,numPolyBasis);
   Wn.SetSize(numDofs,numLocalBasis);
   Vn.SetSize(numDofs,numPolyBasis);
   WV.SetSize(numLocalBasis+numPolyBasis,numLocalBasis+numPolyBasis);
   WnVn.SetSize(numDofs,numLocalBasis+numPolyBasis);

   // build the data matrix
   buildElementRadialBasisMat(el_id,numDofs,dofs_coord,W,Wn);
   buildElementPolyBasisMat(el_id,numDofs,dofs_coord,V,Vn);
   buildWVMat(W,V,WV);
   buildWnVnMat(Wn,Vn,WnVn);
   cout << setprecision(16);
   // if (el_id == 46)
   // {
   //    int b_id;
   //    cout << "element center is: ";
   //    elementCenter[el_id]->Print();
   //    cout << "basis centers loc are:\n";
   //    for (int i = 0; i <selectedBasis[el_id]->Size();i++)
   //    {
   //       b_id =(*selectedBasis[el_id])[i];
   //       cout << "basis " <<  b_id << " : ";
   //       basisCenter[b_id]->Print();
   //    }
   //    cout << "WV is:\n";
   //    //WV.Print(cout,WV.Width());
   //    for (int i = 0; i < WV.Height(); i++)
   //    {
   //       for (int j = 0; j <WV.Width(); j++)
   //       {
   //          cout << WV(i,j) << ' ';
   //       }
   //       cout << '\n';
   //    }
   // }
   
   
   // free the aux variable
   for (int k = 0; k < numDofs; k++)
   {
      delete dofs_coord[k];
   }
}

void RBFSpace::buildDofMat(int el_id, const int numDofs,
                           const FiniteElement *fe,
                           Array<Vector *> &dofs_coord) const
{
   Vector coord(dim);
   ElementTransformation *eltransf = mesh->GetElementTransformation(el_id);
   for (int i = 0; i < numDofs; i++)
   {
      eltransf->Transform(fe->GetNodes().IntPoint(i), coord);
      *dofs_coord[i] = coord;
   }
}


void RBFSpace::solveLocalProlongationMat(const int el_id,
                                         const DenseMatrix &WV,
                                         const DenseMatrix &WnVn,
                                         DenseMatrix &localMat)
{
   int numLocalBasis = selectedBasis[el_id]->Size();
   int numDofs = WnVn.Height();
   DenseMatrix b(numLocalBasis+numPolyBasis,numLocalBasis);
   for (int i = 0; i < numLocalBasis; i++)
   {
      b(i,i) = 1.0;
   }
   DenseMatrixInverse WVinv(WV);
   WVinv.Mult(b,*coef[el_id]);

   // check solve
   // DenseMatrix temp(numLocalBasis+numPolyBasis,numLocalBasis);
   // Mult(WV,*coef[el_id],temp);
   // if (el_id == 46)
   // {
   //    cout << "\nMultiplication results is:\n";
   //    temp.Print(cout,temp.Width());
   // }

   // cout << "solve results is:\n" << setprecision(16);
   // for (int i = 0; i < numLocalBasis+numPolyBasis; i++)
   // {
   //    for (int j = 0; j < numLocalBasis; j++)
   //    {
   //       cout << temp(i,j) << ' ';
   //    }
   //    cout << '\n';
   // }

   // Get Local prolongation matrix
   localMat.SetSize(numDofs,numLocalBasis);
   Mult(WnVn,*coef[el_id],localMat);

   // Solve the coefficient with Lapack
   // for (int i = 0; i < numLocalBasis; i++)
   // {
   //    (*coef[el_id])(i,i) = 1.0;
   // }
   // buildRBFInterpolation(numLocalBasis,numPolyBasis,WV,*coef[el_id]);
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

void RBFSpace::buildElementPolyBasisMat(const int el_id, const int numDofs,
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

void RBFSpace::AssembleProlongationMatrix(const int el_id, const DenseMatrix &localMat)
{
   // element id coresponds to the column indices
   // dofs id coresponds to the row indices
   // the local reconstruction matrix needs to be assembled `vdim` times
   // assume the mesh only contains only 1 type of element
   const Element* el = mesh->GetElement(el_id);
   const FiniteElement *fe = fec->FiniteElementForGeometry(el->GetGeometryType());
   const int numDofs = fe->GetDof();

   int numLocalBasis= selectedBasis[el_id]->Size();
   Array<int> el_dofs;
   Array<int> col_index(numLocalBasis);
   Array<int> row_index(numDofs);

   GetElementVDofs(el_id, el_dofs);
   for(int e = 0; e < numLocalBasis; e++)
   {
      col_index[e] = vdim * (*selectedBasis[el_id])[e];
   }

   for (int v = 0; v < vdim; v++)
   {
      el_dofs.GetSubArray(v * numDofs, numDofs, row_index);
      cP->SetSubMatrix(row_index, col_index, localMat, 1);
      row_index.LoseData();
      // elements id also need to be shift accordingly
      for (int e = 0; e < numLocalBasis; e++)
      {
         col_index[e]++;
      }
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