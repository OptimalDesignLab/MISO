#include "galer_diff.hpp"
#include "utils.hpp"
#include <numeric> 
#include <algorithm> 
#include <vector>
using namespace std;
using namespace mfem;
using namespace mach;

namespace mfem
{

DGDSpace::DGDSpace(Mesh *m, const FiniteElementCollection *f, 
                   Vector center, int degree, int e,
                   int vdim, int ordering, double c, int pl)
   : SpaceType(m, f, vdim, ordering), interpOrder(degree), extra(e),
     basisCenterDummy(center), cond(c), print_level(pl)
{
   dim = m->Dimension();
   numBasis = center.Size()/dim;
   switch(dim)
   {
      case 1: numReqBasis = interpOrder + 1;  break;
      case 2: numReqBasis = (interpOrder+1) * (interpOrder+2) / 2; break;
      case 3: numReqBasis = (interpOrder+1) * (interpOrder+2) * (interpOrder+3) / 6; break;
      default: throw MachException("dim must be 1, 2 or 3.\n");
   }
   
   // initialize member variables
   int nel = GetMesh()->GetNE();
   extraCenter.assign(nel,0);
   selectedBasis.resize(nel);
   sortedEBDistRank.resize(nel);
   selectedElement.resize(numBasis);
   coef.SetSize(nel);
   
   // initialize the stencil/patch based on the given interpolatory order
   InitializeStencil(center);
   
   // build the initial prolongation matrix
   buildProlongationMatrix(center);
   cout << "Check cP size: " << cP->Height() << " x " << cP->Width() << '\n';
   cP_is_set = true;
   // ofstream cp_save("prolong.txt");
	// cP->PrintMatlab(cp_save);
	// cp_save.close();
}

void DGDSpace::InitializeStencil(const Vector &basisCenter)
{
   // declare some intermediate variable
   Vector elemCenter(dim);
   Vector center(dim);
   double dist;
   int i,j,k,bid;
   // clear selected element
   for (i = 0; i < numBasis; i++)
   {
      selectedElement[i].clear(); 
   }
   extraCenter.assign(GetMesh()->GetNE(),0);
   // loop over all the elements to construct the stencil
   vector<size_t> temp;
   vector<double> elementBasisDist;
   for (i = 0; i < GetMesh()->GetNE(); i++)
   {
      elementBasisDist.clear();
      sortedEBDistRank[i].clear();
      selectedBasis[i].clear();
      GetMesh()->GetElementCenter(i,elemCenter);
      // loop over all basis
      for (j = 0; j < numBasis; j++)
      {  
         GetBasisCenter(j,center,basisCenter);
         center -= elemCenter;
         dist = center.Norml2();
         elementBasisDist.push_back(dist);
      }
      // build element/basis stencil based on distance
      sortedEBDistRank[i] = sort_indexes(elementBasisDist);
      for (k = 0;  k < numReqBasis; k++)
      {
         bid = sortedEBDistRank[i][k];
         selectedBasis[i].push_back(bid);
         selectedElement[bid].push_back(i);
      }
   }

   // check if the stencil selection is valid
   for (i = 0; i < numBasis; i++)
   {
      if (selectedElement[i].empty())
      {
         cout << "Basis " << i << " is not selected.\n";
         throw MachException("Basis center is not selected.");
      }
   }
   if (cP) delete cP;
   cP = new mfem::SparseMatrix(GetVSize(),vdim*numBasis);
   adjustCondition = true;
   // cout << "------Basis center local------\n";
   // for (int i = 0; i < numBasis; i++)
   // {  
   //    cout << "basis " << i << ": ";
   //    cout << basisCenter(2*i) << ' ' << basisCenter(2*i+1) << '\n';
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

void DGDSpace::buildProlongationMatrix(const Vector &x)
{
   DenseMatrix V, Vn;
   DenseMatrix localMat;
   for (int i = 0; i < GetMesh()->GetNE(); i++)
   {
      // 1. build basis matrix
      buildDataMat(i,x,V,Vn);

      // 2. build the interpolation matrix
      solveLocalProlongationMat(i,V,Vn,localMat);

      // 3. Assemble prolongation matrix
      AssembleProlongationMatrix(i,localMat);
   }
   cP->Finalize(0);
   adjustCondition = false;
   ofstream cp_save("prolong.txt");
	cP->PrintMatlab(cp_save);
	cp_save.close();
}

void DGDSpace::buildDataMat(int el_id, const Vector &x,
                            DenseMatrix &V, DenseMatrix &Vn)
{
   // Get element dofs coords
   Array<Vector *> dofs_coord;
   GetElementInfo(el_id,dofs_coord);
   // build the data matrix
   buildElementPolyBasisMat(el_id,x,dofs_coord,V,Vn);

   // if V is rank deficit, append more basis
   if (adjustCondition)
   {
      Vector sv;
      V.SingularValues(sv);
      if (sv(0) > cond * sv(numReqBasis-1))
      {
         if (print_level)
            cout << el_id <<  " cond = " << sv(0)/sv(numReqBasis-1) << " (> " << cond << ") ";
      }
      while (sv(0) > cond * sv(numReqBasis-1))
      {
         // build new V/Vn matrices
         addExtraBasis(el_id);
         buildElementPolyBasisMat(el_id,x,dofs_coord,V,Vn);

         // check new condition number
         V.SingularValues(sv);
         if (extraCenter[el_id] > extra) // a tentative cap
         {  
            if (print_level)
               cout << "fail to reduce the V condition number...\n";
            throw MachException("DGDSpace::buildDataMat(): Too much centers added...");
         }
      }
      if (extraCenter[el_id])
      {
         if (print_level)
         {
            cout << " ---> new cond is " << sv(0)/sv(numReqBasis-1) << ", "
                 <<  extraCenter[el_id] << " extra center added.\n";
         }
      }
   }



   // free the aux variable
   for (int k = 0; k < dofs_coord.Size(); k++)
   {
      delete dofs_coord[k];
   }
}

void DGDSpace::addExtraBasis(int el_id)
{
   int numAppendedCenter = extraCenter[el_id];
   int b_id = sortedEBDistRank[el_id][numReqBasis-1+numAppendedCenter+1];
   selectedBasis[el_id].push_back(b_id);
   extraCenter[el_id]++;
   selectedElement[b_id].push_back(el_id);
}

void DGDSpace::buildElementPolyBasisMat(const int el_id,
                                        const Vector &basisCenter,
                                        const Array<Vector *> &dofs_coord,
                                        DenseMatrix &V, DenseMatrix &Vn) const
{
   int i,j,k,l;
   int numDofs = dofs_coord.Size();
   int b_id;
   Vector loc_coord(dim);
   Vector el_center(dim);
   GetMesh()->GetElementCenter(el_id,el_center);
   double vandScale = calcVandScale(el_id,el_center,basisCenter);
   // initialize V, Vn
   int numCenter = selectedBasis[el_id].size();
   V.SetSize(numCenter,numReqBasis);
   Vn.SetSize(numDofs,numReqBasis);
   if (1 == dim)
   {
      double dx;
      // form the V matrix
      for (i = 0; i < numCenter; i++)
      {
         b_id = selectedBasis[el_id][i];
         GetBasisCenter(b_id,loc_coord,basisCenter);
         dx = (loc_coord[0] - el_center[0])/vandScale;
         for (j = 0; j <= interpOrder; j++)
         {
            V(i,j) = pow(dx,j);
         }
      }

      // form the Vn matrix
      for (i = 0; i < numDofs; i++)
      {
         loc_coord = *dofs_coord[i];
         dx = (loc_coord[0] - el_center[0])/vandScale;
         for (j = 0; j <= interpOrder; j++)
         {
            Vn(i,j) = pow(dx,j);
         }
      }
   }
   else if (2 == dim)
   {
      double dx, dy;
      // form the V matrix
      for (i = 0; i < numCenter; i++)
      {
         b_id = selectedBasis[el_id][i];
         GetBasisCenter(b_id,loc_coord,basisCenter);
         dx = (loc_coord[0] - el_center[0])/vandScale;
         dy = (loc_coord[1] - el_center[1])/vandScale;
         int col = 0;
         for (j = 0; j <= interpOrder; j++)
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
         dx = (loc_coord[0] - el_center[0])/vandScale;
         dy = (loc_coord[1] - el_center[1])/vandScale;
         int col = 0;
         for (j = 0; j <= interpOrder; j++)
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
      double dx,dy,dz;
      // form the V matrix
      for (i = 0; i < numCenter; i++)
      {
         b_id = selectedBasis[el_id][i];
         GetBasisCenter(b_id,loc_coord,basisCenter);
         dx = (loc_coord[0] - el_center[0])/vandScale;
         dy = (loc_coord[1] - el_center[1])/vandScale;
         dz = (loc_coord[2] - el_center[2])/vandScale;
         int col = 0;
         for (j = 0; j <= interpOrder; j++)
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
         dx = (loc_coord[0] - el_center[0])/vandScale;
         dy = (loc_coord[1] - el_center[1])/vandScale;
         dz = (loc_coord[2] - el_center[2])/vandScale;
         int col = 0;
         for (j = 0; j <= interpOrder; j++)
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

void DGDSpace::solveLocalProlongationMat(const int el_id,
                                         const DenseMatrix &V,
                                         const DenseMatrix &Vn,
                                         DenseMatrix &localMat) const
{
   int numDofs = Vn.Height();
   int numCenter = selectedBasis[el_id].size();
   DenseMatrix b(numCenter,numCenter);
   b = 0.0;
   for (int i = 0; i < numCenter; i++)
   {
      b(i,i) = 1.0;
   }


   coef[el_id] = new DenseMatrix(numReqBasis,numCenter);

   if (numCenter == numReqBasis)
   {
      DenseMatrixInverse Vinv(V);
      Vinv.Mult(b,*coef[el_id]);
   }
   else
   {
      DenseMatrix Vt(V);
      Vt.Transpose();
      DenseMatrix VtV(numReqBasis,numReqBasis);
      Mult(Vt, V, VtV);

      DenseMatrixInverse Vinv(VtV);
      DenseMatrix Vtb(numReqBasis,numCenter);
      Mult(Vt,b,Vtb);
      Vinv.Mult(Vtb,*coef[el_id]);
   }
   localMat.SetSize(numDofs,numCenter);
   Mult(Vn,*coef[el_id],localMat);

   // check solve
   // if (el_id == 0)
   // {
   //    // cout << "b after solve is:\n";
   //    // b.Print(cout,b.Width());
   //    // for (int i = 0; i < numReqBasis; i++)
   //    // {
   //    //    for (int j = 0; j < numReqBasis; j++)
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
   //    DenseMatrix temp(numReqBasis,numReqBasis);
   //    Mult(V,*coef[el_id],temp);
   //    cout  << "temp results is: \n";
   //    temp.Print(cout,temp.Width());
   // }
   // Get Local prolongation matrix
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

   int numCenter = selectedBasis[el_id].size();
   Array<int> el_dofs;
   Array<int> col_index(numCenter);
   Array<int> row_index(numDofs);

   GetElementVDofs(el_id, el_dofs);
   for(int e = 0; e < numCenter; e++)
   {
      col_index[e] = vdim * selectedBasis[el_id][e];
   }
   for (int v = 0; v < vdim; v++)
   {
      el_dofs.GetSubArray(v * numDofs, numDofs, row_index);
      cP->SetSubMatrix(row_index, col_index, localMat, 0);
      // row_index.LoseData();
      // elements id also need to be shift accordingly
      for (int e = 0; e < numCenter; e++)
      {
         col_index[e]++;
      }
   }
}

void DGDSpace::GetdPdc(const int id, const Vector &basisCenter,
                       SparseMatrix &dpdc)
{
   int xyz = id % dim; // determine whether it is x, y, or z
   int b_id = id / dim; // determine the basis id

   int numLocalElem = selectedElement[b_id].size();
   int el_id;
   DenseMatrix V;
   DenseMatrix dV;
   DenseMatrix Vn;
   DenseMatrix dVn;
   DenseMatrix dpdc_block;
   int numCenter;
   for (int i = 0; i < numLocalElem; i++)
   {
      el_id = selectedElement[b_id][i];
      numCenter = selectedBasis[el_id].size();
      buildDerivDataMat(el_id,b_id,xyz,basisCenter,V,dV,Vn,dVn);
      dpdc_block.SetSize(Vn.Height(),numCenter);
      
      // cout << "Element id is " << el_id << '\n';
      // cout << "element center is: ";
      // Vector el_center(dim);
      // GetMesh()->GetElementCenter(el_id,el_center);
      // el_center.Print();

      // cout << "selected basis are: ";
      // selectedBasis[el_id]->Print(cout,selectedBasis[el_id]->Size());
      // int ii;
      // for (int l = 0; l < selectedBasis[el_id]->Size(); l++)
      // {
      //    ii = (*selectedBasis[el_id])[l];
      //    GetBasisCenter(ii,el_center);
      //    el_center.Print();
      // }
      // cout << "check V: \n";
      // V.Print(cout,V.Width());
      // cout << "check dV:\n";
      // dV.Print(cout,dV.Width());
      // cout << "check Vn:\n";
      // Vn.Print(cout,Vn.Width());

      // ofstream v_save("v.txt");
      // V.PrintMatlab(v_save);
      // v_save.close();

      // ofstream dv_save("dv.txt");
      // dV.PrintMatlab(dv_save);
      // dv_save.close();

      // ofstream vn_save("vn.txt");
      // Vn.PrintMatlab(vn_save);
      // vn_save.close();

      // ofstream coef_save("coef.txt");
      // coef[el_id]->PrintMatlab(coef_save);
      // coef_save.close();


      // V is a square matrix
      if (numReqBasis == numCenter)
      {
         DenseMatrix temp_mat1(numReqBasis);
         DenseMatrix temp_mat2(numReqBasis);
         Mult(dV,*coef[el_id],temp_mat1);
         Mult(*coef[el_id],temp_mat1,temp_mat2); 
         temp_mat2.Neg();  // -V^-1 * dV * V^-1
         Mult(Vn,temp_mat2,dpdc_block); //  dpdc = Vn * temp2
      }
      // V is overdetermined
      else
      {
         DenseMatrix Vt(V);
         Vt.Transpose(); // get V^t

         DenseMatrix vtv(numReqBasis); 
         Mult(Vt,V,vtv);
         DenseMatrixInverse vtvinv(vtv); // get (V^t V)^-1

         DenseMatrix dVt(dV);
         dVt.Transpose(); // get dV^t

         DenseMatrix dvtv(numReqBasis);
         Mult(Vt,dV,dvtv);
         AddMult(dVt,V,dvtv); // compute d V^tV / dc

         DenseMatrix temp_mat1(numReqBasis);
         DenseMatrix deriv_p1(numReqBasis,numCenter);
         vtvinv.Mult(dvtv,temp_mat1);
         Mult(temp_mat1,*coef[el_id],deriv_p1);
         deriv_p1.Neg(); // first part of the derivatve

         DenseMatrix deriv_p2(numReqBasis,numCenter);
         vtvinv.Mult(dVt,deriv_p2);
         
         deriv_p1 += deriv_p2;
         Mult(Vn,deriv_p1,dpdc_block);
      }
      // assemble is back to the derivative matrix
      AssembleDerivMatrix(el_id,dpdc_block,dpdc);
   }
   dpdc.Finalize(0);
}

void DGDSpace::buildDerivDataMat(const int el_id, const int b_id, const int xyz,
                                 const Vector &basisCenter,
                                 DenseMatrix &V, DenseMatrix &dV,
                                 DenseMatrix &Vn, DenseMatrix &dVn) const
{
   // get element related data
   Array<Vector *> dofs_coord;
   GetElementInfo(el_id, dofs_coord);
   // build the data matrix
   buildElementDerivMat(el_id,b_id,basisCenter,xyz,dofs_coord,dV,dVn);
   buildElementPolyBasisMat(el_id,basisCenter,dofs_coord,V,Vn);
   // free the aux variable
   for (int k = 0; k < dofs_coord.Size(); k++)
   {
      delete dofs_coord[k];
   }
}

void DGDSpace::buildElementDerivMat(const int el_id, const int b_id,
                                    const Vector &basisCenter,
                                    const int xyz,
                                    const Array<Vector*> &dofs_coord,
                                    DenseMatrix &dV,
                                    DenseMatrix &dVn) const
{
   int i,j,k,col;
   double dx,dy,dz;
   const int numCenter = selectedBasis[el_id].size();
   const int numDofs = dofs_coord.Size();

   auto it = find(selectedBasis[el_id].begin(),selectedBasis[el_id].end(),b_id);
   const int row_idx = it - selectedBasis[el_id].begin();

   Vector loc_coord(dim);
   GetBasisCenter(b_id,loc_coord,basisCenter);

   Vector el_center(dim);
   GetMesh()->GetElementCenter(el_id,el_center);
   double vandScale = calcVandScale(el_id,el_center,basisCenter);

   dV.SetSize(numCenter,numReqBasis);
   dVn.SetSize(numDofs,numReqBasis);

   dV  = 0.0;
   dVn = 0.0;
   col = 1;
   if (1 == dim)
   {
      // form the dV matrix (only one row needs update)
      dx = loc_coord[0] - el_center[0];
      dV(row_idx,0) = 0.0;
      for (j = 1; j <= interpOrder; j++)
      {
         dV(row_idx,j) = j * pow(dx,j-1) / pow(vandScale,j);
      }
   }
   else if (2 == dim)
   {
      // form the dV matrix
      dx = loc_coord[0] - el_center[0];
      dy = loc_coord[1] - el_center[1];
      dV(row_idx,0) = 0.0;
      for (j = 1; j <= interpOrder; j++)
      {
         for (k = 0; k <= j; k ++)
         {
            if (0 == k)
            {
               dV(row_idx,col) = (0 == xyz) ? j * pow(dx,j-1) / pow(vandScale,j) : 0.0;
            }
            else if(j == k)
            {
               dV(row_idx,col) = (0 == xyz)? 0.0 : k * pow(dy,k-1) / pow(vandScale,j);
            }
            else
            {
               dV(row_idx,col) = (0 == xyz) ? (j-k) * pow(dx,j-k-1) * pow(dy,k) / pow(vandScale,j)
                              : pow(dx,j-k) * k * pow(dy,k-1)/pow(vandScale,j);
            }
            col++;
         }
      }
   }
}



void DGDSpace::AssembleDerivMatrix(const int el_id, const DenseMatrix &localMat,
                                   SparseMatrix &dpdc) const
{
   // element id coresponds to the column indices
   // dofs id coresponds to the row indices
   // the local reconstruction matrix needs to be assembled `vdim` times
   // assume the mesh only contains only 1 type of element
   const Element* el = mesh->GetElement(el_id);
   const FiniteElement *fe = fec->FiniteElementForGeometry(el->GetGeometryType());
   const int numDofs = fe->GetDof();

   int numCenter= selectedBasis[el_id].size();
   Array<int> el_dofs;
   Array<int> col_index(numCenter);
   Array<int> row_index(numDofs);

   GetElementVDofs(el_id,el_dofs);
   for(int e = 0; e < numCenter; e++)
   {
      col_index[e] = vdim * selectedBasis[el_id][e];
   }

   for (int v = 0; v < vdim; v++)
   {
      el_dofs.GetSubArray(v * numDofs, numDofs, row_index);
      dpdc.SetSubMatrix(row_index, col_index, localMat, 0);
      // elements id also need to be shift accordingly
      for (int e = 0; e < numCenter; e++)
      {
         col_index[e]++;
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

void DGDSpace::GetBasisCenter(const int b_id, Vector &center,
                              const Vector &basisCenter) const
{
   for (int i = 0; i < dim; i++)
   {
      center(i) = basisCenter(b_id*dim+i);
   }
}

void DGDSpace::GetElementInfo(int el_id, Array<Vector *> &dofs_coord) const
{
   const Element *el = mesh->GetElement(el_id);
   const FiniteElement *fe = fec->FiniteElementForGeometry(el->GetGeometryType());
   const int numDofs = fe->GetDof();
   ElementTransformation *eltransf = mesh->GetElementTransformation(el_id);
   
   // get the dofs coord
   dofs_coord.SetSize(numDofs);
   Vector coord(dim);
   for (int k = 0; k <numDofs; k++)
   {
      dofs_coord[k] = new Vector(dim);
      eltransf->Transform(fe->GetNodes().IntPoint(k), coord);
      *dofs_coord[k] = coord;
   }
}

double DGDSpace::calcVandScale(const int el_id, 
                               const Vector &el_center,
                               const Vector &basisCenter) const
{
   //get the most furthe basis 
   int numCenter = selectedBasis[el_id].size();
   int bid = selectedBasis[el_id][numCenter-1];
   Vector center(dim);
   GetBasisCenter(bid,center,basisCenter);
   // compute the scale
   center -= el_center;
   return center.Norml2();
}

DGDSpace::~DGDSpace()
{
   for (int k = 0; k < GetMesh()->GetNE(); k++)
   {
      delete coef[k];
   }
}

} // end of namespace mfem