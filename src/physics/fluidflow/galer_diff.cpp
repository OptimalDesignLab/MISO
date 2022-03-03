#include "galer_diff.hpp"
#include "utils.hpp"
#include <numeric> 
#include <algorithm> 

using namespace std;
using namespace mfem;
using namespace mach;

namespace mfem
{

DGDSpace::DGDSpace(Mesh *m, const FiniteElementCollection *f, 
                   Array<Vector*> center, int degree, int e,
                   int vdim, int ordering)
   : SpaceType(m, f, vdim, ordering), basisCenter(center), polyOrder(degree),
     extra(e)
{
   // numBasis should not be greater than the number of elements
   dim = m->Dimension();
   numBasis = center.Size();

   switch(dim)
   {
      case 1: numPolyBasis = polyOrder + 1;  break;
      case 2: numPolyBasis = (polyOrder+1) * (polyOrder+2) / 2; break;
      case 3: numPolyBasis = (polyOrder+1) * (polyOrder+2) * (polyOrder+3) / 6; break;
      default: throw MachException("dim must be 1, 2 or 3.\n");
   }
   numLocalBasis = numPolyBasis + extra;
   cout << "Number of required polynomial basis is " << numPolyBasis << '\n';
   cout << "Number of element local basis is " << numLocalBasis << '\n';

   // initialize the stencil/patch
   InitializeStencil();

   // initialize the prolongation matrix
   cP = new mfem::SparseMatrix(GetVSize(),vdim*numBasis);
   buildProlongation();
   cP->Finalize();
   cP_is_set = true;
   cout << "Check cP size: " << cP->Height() << " x " << cP->Width() << '\n';
   ofstream cp_save("ProlongationMat.txt");
   cP->PrintMatlab(cp_save);
   cp_save.close();

}

void DGDSpace::InitializeStencil()
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
      coef[i] = new DenseMatrix(numPolyBasis,numLocalBasis);
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
      for (int j = 0; j < numLocalBasis; j++)
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

void DGDSpace::buildProlongation() const
{
   // declare soma matrix variables
   DenseMatrix V, Vn;
   DenseMatrix localMat;
   // loop over element to build local and global prolongation matrix
   for (int i = 0; i < GetMesh()->GetNE(); i++)
   {
      // 1. build basis matrix
      buildDataMat(i,V,Vn);
      // if (i == 0)
      // {
      //    cout << "Check V:\n";
      //    for (int i = 0; i < V.Height(); ++i)
      //    {
      //       for (int j = 0; j < V.Width(); j++)
      //       {
      //          cout << V(i,j) << ' ';
      //       }
      //       cout << '\n';
      //    }
      // }
      // 2. build the interpolation matrix
      solveLocalProlongationMat(i,V,Vn,localMat);

      // 3. Assemble prolongation matrix
      AssembleProlongationMatrix(i,localMat);
   }
}

void DGDSpace::buildDataMat(int el_id, DenseMatrix &V,
                            DenseMatrix &Vn) const
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

   V.SetSize(numLocalBasis,numPolyBasis);
   Vn.SetSize(numDofs,numPolyBasis);

   // build the data matrix
   buildElementPolyBasisMat(el_id,numDofs,dofs_coord,V,Vn);
   
   // free the aux variable
   for (int k = 0; k < numDofs; k++)
   {
      delete dofs_coord[k];
   }
}

void DGDSpace::solveLocalProlongationMat(const int el_id,
                                         const DenseMatrix &V,
                                         const DenseMatrix &Vn,
                                         DenseMatrix &localMat) const
{
   int numDofs = Vn.Height();
   DenseMatrix b(numLocalBasis,numLocalBasis);
   b = 0.0;
   for (int i = 0; i < numLocalBasis; i++)
   {
      b(i,i) = 1.0;
   }
   //buildDGDInterpolation(numLocalBasis,numPolyBasis,V,b);

   if (numPolyBasis == numLocalBasis)
   {
      DenseMatrixInverse Vinv(V);
      Vinv.Mult(b,*coef[el_id]);
   }
   else
   {
      DenseMatrix Vt(V);
      Vt.Transpose();
      DenseMatrix VtV(numPolyBasis,numPolyBasis);
      Mult(Vt, V, VtV);

      DenseMatrixInverse Vinv(VtV);
      DenseMatrix Vtb(numPolyBasis,numLocalBasis);
      Mult(Vt,b,Vtb);
      Vinv.Mult(Vtb,*coef[el_id]);
   }
   

   // check solve
   // if (el_id == 0)
   // {
   //    // cout << "b after solve is:\n";
   //    // b.Print(cout,b.Width());
   //    // for (int i = 0; i < numPolyBasis; i++)
   //    // {
   //    //    for (int j = 0; j < numLocalBasis; j++)
   //    //    {
   //    //       (*coef[el_id])(i,j) = b(i,j);
   //    //    }
   //    // }
   //    cout << "coef is:\n" << setprecision(16);
   //    for (int i = 0; i < coef[el_id]->Height(); ++i)
   //    {
   //       for (int j = 0; j < coef[el_id]->Width(); j++)
   //       {
   //          cout << (*coef[el_id])(i,j) << ' ';
   //       }
   //       cout << '\n';
   //    }
   //    DenseMatrix temp(numLocalBasis,numLocalBasis);
   //    Mult(V,*coef[el_id],temp);
   //    cout  << "temp results is: \n";
   //    temp.Print(cout,temp.Width());
   // }
   // Get Local prolongation matrix
   localMat.SetSize(numDofs,numLocalBasis);
   Mult(Vn,*coef[el_id],localMat);
}


void DGDSpace::buildElementPolyBasisMat(const int el_id, const int numDofs,
                                        const Array<Vector *> &dofs_coord,
                                        DenseMatrix &V, DenseMatrix &Vn) const
{
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

DGDSpace::~DGDSpace()
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

void DGDSpace::AssembleProlongationMatrix(const int el_id, const DenseMatrix &localMat) const
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

void DGDSpace::GetdPdc(const int b_id, SparseMatrix &dpdc)
{
   int numLocalElem = selectedElement[b_id]->Size();
   int el_id;
   DenseMatrix dV;
   DenseMatrix Vn;
   DenseMatrix dpdc_block;
   for (int i = 0; i < numLocalElem; i++)
   {
      el_id = (*selectedBasis[b_id])(i);
      buildDerivDataMat(el_id,b_id,dV,Vn);
      dpdc_block.SetSize(Vn.Height(),numLocalBasis);
      // V is a square matrix
      if (numPolyBasis == numLocalBasis)
      {
         DenseMatrix temp_mat1(numLocalBasis);
         DenseMatrix temp_mat2(numLocalBasis);
         Mult(dV,*coef[el_id],temp_mat1);
         Mult(*coef[el_id],temp_mat1,temp_mat2); 
         temp_mat2.Neg();  // -V^-1 * dV * V^-1
         Mult(Vn,temp_mat2,dpdc_block); //  dpdc = Vn * temp2
      }
      // V is overdetermined
      else
      {

      }

      // assemble is back to the derivative matrix

   }
}

void DGDSpace::buildDerivDataMat(const int el_id, const int b_id,
                                 DenseMatrix &dV, DenseMatrix &Vn) const
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

   dV.SetSize(numLocalBasis,numPolyBasis);
   Vn.SetSize(numDofs,numPolyBasis);

   // build the data matrix
   buildElementDerivMat(el_id,numDofs,dofs_coord,dV,Vn);
   
   // free the aux variable
   for (int k = 0; k < numDofs; k++)
   {
      delete dofs_coord[k];
   }
}

void DGDSpace::buildElementDerivMat(const int el_id, const int numDofs,
                                    const Array<mfem::Vector*> &dofs_coord,
                                    DenseMatrix &dV,
                                    DenseMatrix &Vn) const;
{
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

vector<size_t> DGDSpace::sort_indexes(const vector<double> &v)
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