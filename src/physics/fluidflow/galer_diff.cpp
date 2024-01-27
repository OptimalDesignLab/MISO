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
                      Vector center, int degree, int e,
                      int vdim, int ordering)
       : SpaceType(m, f, vdim, ordering), polyOrder(degree), extra(e),
         basisCenterDummy(center)
   {
      dim = m->Dimension();
      numBasis = center.Size() / dim;
      switch (dim)
      {
      case 1:
         numPolyBasis = polyOrder + 1;
         break;
      case 2:
         numPolyBasis = (polyOrder + 1) * (polyOrder + 2) / 2;
         break;
      case 3:
         numPolyBasis = (polyOrder + 1) * (polyOrder + 2) * (polyOrder + 3) / 6;
         break;
      default:
         throw MachException("dim must be 1, 2 or 3.\n");
      }
      numLocalBasis = numPolyBasis + extra;
      cout << "Number of total basis center is " << center.Size() / dim << '\n';
      cout << "Number of required polynomial basis is " << numPolyBasis << '\n';
      cout << "Number of element local basis is " << numLocalBasis << '\n';

      // initialize the stencil/patch
      InitializeStencil(center);

      // build the initial prolongation matrix
      cP = new mfem::SparseMatrix(GetVSize(), vdim * numBasis);
      cP_is_set = true;
      buildProlongationMatrix(center);
      cout << "Check cP size: " << cP->Height() << " x " << cP->Width() << '\n';
   }

   void DGDSpace::InitializeStencil(const Vector &basisCenter)
   {
      int i, j, k;
      int num_el = mesh->GetNE();

      // initialize the all element centers for later used
      elementBasisDist.assign(num_el, {});
      selectedBasis.assign(num_el, {});
      selectedElement.assign(numBasis, {});
      coef.SetSize(GetMesh()->GetNE());

      // aux data
      double dist;
      Vector elemCenter(dim);
      Vector center(dim);
      // loop over all the elements to construct the stencil
      vector<size_t> temp;
      for (i = 0; i < num_el; i++)
      {
         coef[i] = new DenseMatrix(numPolyBasis, numLocalBasis);
         mesh->GetElementCenter(i, elemCenter);
         // loop over all basis
         for (j = 0; j < numBasis; j++)
         {
            GetBasisCenter(j, center, basisCenter);
            center -= elemCenter;
            dist = center.Norml2();
            elementBasisDist[i].push_back(dist);
         }
         // build element/basis stencil based on distance
         temp = sort_indexes(elementBasisDist[i]);
         for (k = 0; k < numLocalBasis; k++)
         {
            selectedBasis[i].push_back(temp[k]);
            selectedElement[temp[k]].push_back(i);
         }
      }

      cout << "------Check the stencil------\n";
      cout << "------Basis center loca------\n";
      for (int i = 0; i < numBasis; i++)
      {
         cout << "basis " << i << ": ";
         cout << basisCenter(2 * i) << ' ' << basisCenter(2 * i + 1) << '\n';
      }
      cout << '\n';
      cout << "------Elem's  stencil------\n";
      for (int i = 0; i < num_el; i++)
      {
         cout << "Element " << i << ": ";
         for (int j = 0; j < selectedBasis[i].size(); j++)
         {
            cout << selectedBasis[i][j] << ' ';
         }
         cout << '\n';
      }
      cout << '\n';
      cout << "------Basis's  element------\n";
      for (int k = 0; k < numBasis; k++)
      {
         cout << "basis " << k << ": ";
         for (int l = 0; l < selectedElement[k].size(); l++)
         {
            cout << selectedElement[k][l] << ' ';
         }
         cout << '\n';
      }
   }

   void DGDSpace::GetBasisCenter(const int b_id, Vector &center,
                                 const Vector &basisCenter) const
   {
      for (int i = 0; i < dim; i++)
      {
         center(i) = basisCenter(b_id * dim + i);
      }
   }

   void DGDSpace::buildProlongationMatrix(const Vector &x)
   {
      DenseMatrix V, Vn;
      DenseMatrix localMat;
      int num_el = mesh->GetNE();
      for (int i = 0; i < num_el; ++i)
      {
         // 1. build basis matrix
         buildDataMat(i, x, V, Vn);

         // 2. build the interpolation matrix
         solveLocalProlongationMat(i, V, Vn, localMat);

         // 3. Assemble prolongation matrix
         AssembleProlongationMatrix(i, localMat);
      }
      cP->Finalize();
      // ofstream cp_save("prolong.txt");
      // cP->PrintMatlab(cp_save);
      // cp_save.close();
   }

   void DGDSpace::buildDataMat(int el_id, const Vector &x,
                               DenseMatrix &V, DenseMatrix &Vn) const
   {
      // get element related data
      const Element *el = mesh->GetElement(el_id);
      const FiniteElement *fe = fec->FiniteElementForGeometry(el->GetGeometryType());
      const int numDofs = fe->GetDof();
      ElementTransformation *eltransf = mesh->GetElementTransformation(el_id);

      // get the dofs coord
      Array<Vector> dofs_coord;
      // std::vector<std::vector<double>> dofs_coord;
      dofs_coord.SetSize(numDofs);
      Vector coord(dim);
      for (int k = 0; k < numDofs; k++)
      {
         dofs_coord[k].SetSize(dim);
         eltransf->Transform(fe->GetNodes().IntPoint(k), dofs_coord[k]);
      }

      V.SetSize(numLocalBasis, numPolyBasis);
      Vn.SetSize(numDofs, numPolyBasis);
      // build the data matrix
      buildElementPolyBasisMat(el_id, x, numDofs, dofs_coord, V, Vn);

      // free the aux variable
      // for (int k = 0; k < numDofs; k++)
      // {
      //    delete dofs_coord[k];
      // }
   }

   void DGDSpace::solveLocalProlongationMat(const int el_id,
                                            const DenseMatrix &V,
                                            const DenseMatrix &Vn,
                                            DenseMatrix &localMat) const
   {
      int numDofs = Vn.Height();
      DenseMatrix b(numLocalBasis, numLocalBasis);
      b = 0.0;
      for (int i = 0; i < numLocalBasis; i++)
      {
         b(i, i) = 1.0;
      }
      // buildDGDInterpolation(numLocalBasis,numPolyBasis,V,b);

      if (numPolyBasis == numLocalBasis)
      {
         DenseMatrixInverse Vinv(V);
         Vinv.Mult(b, *coef[el_id]);
      }
      else
      {
         DenseMatrix Vt(V);
         Vt.Transpose();
         DenseMatrix VtV(numPolyBasis, numPolyBasis);
         Mult(Vt, V, VtV);

         DenseMatrixInverse Vinv(VtV);
         DenseMatrix Vtb(numPolyBasis, numLocalBasis);
         Mult(Vt, b, Vtb);
         Vinv.Mult(Vtb, *coef[el_id]);
      }
      localMat.SetSize(numDofs, numLocalBasis);
      Mult(Vn, *coef[el_id], localMat);

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
   }

   const std::vector<int> &DGDSpace::GetSelectedBasis(int el_id)
   {
      return selectedBasis[el_id];
   }

   const std::vector<int> &DGDSpace::GetSelectedElement(int b_id)
   {
      return selectedElement[b_id];
   }

   void DGDSpace::buildElementPolyBasisMat(const int el_id,
                                           const Vector &basisCenter,
                                           const int numDofs,
                                           const Array<Vector> &dofs_coord,
                                           DenseMatrix &V, DenseMatrix &Vn) const
   {
      int i, j, k, l;
      double dx, dy, dz;
      int b_id;
      Vector loc_coord(dim);
      Vector el_center(dim);
      mesh->GetElementCenter(el_id, el_center);
      if (1 == dim)
      {
         // form the V matrix
         for (i = 0; i < numLocalBasis; i++)
         {
            b_id = selectedBasis[el_id][i];
            GetBasisCenter(b_id, loc_coord, basisCenter);
            dx = loc_coord[0] - el_center[0];
            for (j = 0; j <= polyOrder; j++)
            {
               V(i, j) = pow(dx, j);
            }
         }

         // form the Vn matrix
         for (i = 0; i < numDofs; i++)
         {
            loc_coord = dofs_coord[i];
            dx = loc_coord[0] - el_center[0];
            for (j = 0; j <= polyOrder; j++)
            {
               Vn(i, j) = pow(dx, j);
            }
         }
      }
      else if (2 == dim)
      {
         // form the V matrix
         for (i = 0; i < numLocalBasis; i++)
         {
            b_id = selectedBasis[el_id][i];
            GetBasisCenter(b_id, loc_coord, basisCenter);
            dx = loc_coord[0] - el_center[0];
            dy = loc_coord[1] - el_center[1];
            int col = 0;
            for (j = 0; j <= polyOrder; j++)
            {
               for (k = 0; k <= j; k++)
               {
                  V(i, col) = pow(dx, j - k) * pow(dy, k);
                  col++;
               }
            }
         }
         // form the Vn matrix
         for (i = 0; i < numDofs; i++)
         {
            loc_coord = dofs_coord[i];
            dx = loc_coord[0] - el_center[0];
            dy = loc_coord[1] - el_center[1];
            int col = 0;
            for (j = 0; j <= polyOrder; j++)
            {
               for (k = 0; k <= j; k++)
               {
                  Vn(i, col) = pow(dx, j - k) * pow(dy, k);
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
            b_id = selectedBasis[el_id][i];
            GetBasisCenter(b_id, loc_coord, basisCenter);
            dx = loc_coord[0] - el_center[0];
            dy = loc_coord[1] - el_center[1];
            dz = loc_coord[2] - el_center[2];
            int col = 0;
            for (j = 0; j <= polyOrder; j++)
            {
               for (k = 0; k <= j; k++)
               {
                  for (l = 0; l <= j - k; l++)
                  {
                     V(i, col) = pow(dx, j - k - l) * pow(dy, l) * pow(dz, k);
                     col++;
                  }
               }
            }
         }

         // form the Vn matrix
         for (i = 0; i < numDofs; i++)
         {
            loc_coord = dofs_coord[i];
            dx = loc_coord[0] - el_center[0];
            dy = loc_coord[1] - el_center[1];
            dz = loc_coord[2] - el_center[2];
            int col = 0;
            for (j = 0; j <= polyOrder; j++)
            {
               for (k = 0; k <= j; k++)
               {
                  for (l = 0; l <= j - k; l++)
                  {
                     Vn(i, col) = pow(dx, j - k - l) * pow(dy, l) * pow(dz, k);
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
         delete coef[k];
      }
   }

   void DGDSpace::AssembleProlongationMatrix(const int el_id, const DenseMatrix &localMat) const
   {
      // element id coresponds to the column indices
      // dofs id coresponds to the row indices
      // the local reconstruction matrix needs to be assembled `vdim` times
      // assume the mesh only contains only 1 type of element
      const Element *el = mesh->GetElement(el_id);
      const FiniteElement *fe = fec->FiniteElementForGeometry(el->GetGeometryType());
      const int numDofs = fe->GetDof();

      int numLocalBasis = selectedBasis[el_id].size();
      Array<int> el_dofs;
      Array<int> col_index(numLocalBasis);
      Array<int> row_index(numDofs);

      GetElementVDofs(el_id, el_dofs);
      for (int e = 0; e < numLocalBasis; e++)
      {
         col_index[e] = vdim * selectedBasis[el_id][e];
      }
      for (int v = 0; v < vdim; v++)
      {
         el_dofs.GetSubArray(v * numDofs, numDofs, row_index);
         cP->SetSubMatrix(row_index, col_index, localMat, 1);
         // row_index.LoseData();
         //  elements id also need to be shift accordingly
         for (int e = 0; e < numLocalBasis; e++)
         {
            col_index[e]++;
         }
      }
   }

   void DGDSpace::GetdPdc(const int id, const Vector &basisCenter,
                          SparseMatrix &dpdc)
   {
      int xyz = id % dim;  // determine whether it is x, y, or z
      int b_id = id / dim; // determine the basis id

      int numLocalElem = selectedElement[b_id].size();
      int el_id;
      DenseMatrix V;
      DenseMatrix dV;
      DenseMatrix Vn;
      DenseMatrix dpdc_block;
      // cout << "Selected elements are: ";
      // selectedElement[b_id]->Print(cout,numLocalElem);
      for (int i = 0; i < numLocalElem; i++)
      {
         cout << "Element " << i << ":\n";
         el_id = selectedElement[b_id][i];
         buildDerivDataMat(el_id, b_id, xyz, basisCenter, V, dV, Vn);
         dpdc_block.SetSize(Vn.Height(), numLocalBasis);
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
         if (numPolyBasis == numLocalBasis)
         {
            DenseMatrix temp_mat1(numLocalBasis);
            DenseMatrix temp_mat2(numLocalBasis);
            Mult(dV, *coef[el_id], temp_mat1);
            Mult(*coef[el_id], temp_mat1, temp_mat2);
            temp_mat2.Neg();                 // -V^-1 * dV * V^-1
            Mult(Vn, temp_mat2, dpdc_block); //  dpdc = Vn * temp2
         }
         // V is overdetermined
         else
         {
            DenseMatrix Vt(V);
            Vt.Transpose(); // get V^t

            DenseMatrix vtv(numPolyBasis);
            Mult(Vt, V, vtv);
            DenseMatrixInverse vtvinv(vtv); // get (V^t V)^-1

            DenseMatrix dVt(dV);
            dVt.Transpose(); // get dV^t

            DenseMatrix dvtv(numPolyBasis);
            Mult(Vt, dV, dvtv);
            AddMult(dVt, V, dvtv); // compute d V^tV / dc

            DenseMatrix temp_mat1(numPolyBasis);
            DenseMatrix deriv_p1(numPolyBasis, numLocalBasis);
            vtvinv.Mult(dvtv, temp_mat1);
            Mult(temp_mat1, *coef[el_id], deriv_p1);
            deriv_p1.Neg(); // first part of the derivatve

            DenseMatrix deriv_p2(numPolyBasis, numLocalBasis);
            vtvinv.Mult(dVt, deriv_p2);

            deriv_p1 += deriv_p2;
            Mult(Vn, deriv_p1, dpdc_block);
         }
         cout << "assemble back to dpdc\n";
         // assemble is back to the derivative matrix
         AssembleDerivMatrix(el_id, dpdc_block, dpdc);
      }
      dpdc.Finalize();
   }

   void DGDSpace::buildDerivDataMat(const int el_id, const int b_id, const int xyz,
                                    const Vector &basisCenter,
                                    DenseMatrix &V, DenseMatrix &dV,
                                    DenseMatrix &Vn) const
   {
      // get element related data
      const Element *el = mesh->GetElement(el_id);
      const FiniteElement *fe = fec->FiniteElementForGeometry(el->GetGeometryType());
      const int numDofs = fe->GetDof();
      ElementTransformation *eltransf = mesh->GetElementTransformation(el_id);

      // get the dofs coord
      Array<Vector> dofs_coord;
      dofs_coord.SetSize(numDofs);
      Vector coord(dim);
      for (int k = 0; k < numDofs; k++)
      {
         dofs_coord[k].SetSize(dim);
         eltransf->Transform(fe->GetNodes().IntPoint(k), dofs_coord[k]);
      }

      V.SetSize(numLocalBasis, numPolyBasis);
      dV.SetSize(numLocalBasis, numPolyBasis);
      Vn.SetSize(numDofs, numPolyBasis);

      // build the data matrix
      std::cout << "building  dV, ";
      buildElementDerivMat(el_id, b_id, basisCenter, xyz, numDofs, dofs_coord, dV);
      std::cout << "building V, ";
      buildElementPolyBasisMat(el_id, basisCenter, numDofs, dofs_coord, V, Vn);
      // free the aux variable
      // for (int k = 0; k < numDofs; k++)
      // {
      //    delete dofs_coord[k];
      // }
   }

   void DGDSpace::buildElementDerivMat(const int el_id, const int b_id,
                                       const Vector &basisCenter,
                                       const int xyz, const int numDofs,
                                       const Array<Vector> &dofs_coord,
                                       DenseMatrix &dV) const
   {
      int j, k, col;
      double dx, dy;
      const auto &basis_list = selectedBasis[el_id];
      auto itr = std::find(basis_list.begin(), basis_list.end(), b_id);
      const int row_idx = std::distance(basis_list.begin(), itr);
      Vector loc_coord(dim);
      Vector el_center(dim);
      GetBasisCenter(b_id, loc_coord, basisCenter);
      mesh->GetElementCenter(el_id, el_center);
      dV = 0.0;
      col = 1;
      if (1 == dim)
      {
         // form the dV matrix (only one row needs update)
         dx = loc_coord[0] - el_center[0];
         dV(row_idx, 0) = 0.0;
         for (j = 1; j <= polyOrder; j++)
         {
            dV(row_idx, j) = j * pow(dx, j - 1);
         }
      }
      else if (2 == dim)
      {
         // form the dV matrix
         dx = loc_coord[0] - el_center[0];
         dy = loc_coord[1] - el_center[1];
         dV(row_idx, 0) = 0.0;
         for (j = 1; j <= polyOrder; j++)
         {
            for (k = 0; k <= j; k++)
            {
               if (0 == k)
               {
                  dV(row_idx, col) = (0 == xyz) ? j * pow(dx, j - 1) : 0.0;
               }
               else if (j == k)
               {
                  dV(row_idx, col) = (0 == xyz) ? 0.0 : k * pow(dy, k - 1);
               }
               else
               {
                  dV(row_idx, col) = (0 == xyz) ? (j - k) * pow(dx, j - k - 1) * pow(dy, k)
                                                : pow(dx, j - k) * k * pow(dy, k - 1);
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
      const Element *el = mesh->GetElement(el_id);
      const FiniteElement *fe = fec->FiniteElementForGeometry(el->GetGeometryType());
      const int numDofs = fe->GetDof();

      int numLocalBasis = selectedBasis[el_id].size();
      Array<int> el_dofs;
      Array<int> col_index(numLocalBasis);
      Array<int> row_index(numDofs);

      GetElementVDofs(el_id, el_dofs);
      for (int e = 0; e < numLocalBasis; e++)
      {
         col_index[e] = vdim * selectedBasis[el_id][e];
      }

      for (int v = 0; v < vdim; v++)
      {
         el_dofs.GetSubArray(v * numDofs, numDofs, row_index);
         dpdc.SetSubMatrix(row_index, col_index, localMat, 1);
         // elements id also need to be shift accordingly
         for (int e = 0; e < numLocalBasis; e++)
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
                  [&v](size_t i1, size_t i2)
                  { return v[i1] < v[i2]; });

      return idx;
   }

} // end of namespace mfem