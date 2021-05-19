#include "rbfspace.hpp"
#include "sbp_fe.hpp"
#include "utils.hpp"
#include <fstream>
#include <iostream>

using namespace std;
using namespace mfem;
using namespace mach;

namespace mfem
{

RBFSpace::RBFSpace(Mesh *m, const FiniteElementCollection *f, int nb,
                   int vdim, double span_coef, int ordering)
   : FiniteElementSpace(m, f, vdim, ordering)
{
   dim = mesh->Dimension();
   num_basis = nb;

   if (1 == dim)
   {
      shape_param = 1;
   }
   else if (2 == dim)
   {
      shape_param = 3;
   }
   else if (3 == dim)
   {
      shape_param = 6;
   }

   lam.SetSize(shape_param, num_basis);
   basis_center.SetSize(dim, num_basis);
   lam = 0.0;
   // initialize the basis centers and span
   int geom;
   ElementTransformation *eltransf;
   Vector diff(dim), cent(dim);
   double h = 0.0;

   for (int i = 0; i < num_basis; i++)
   {
      switch (dim)
      {
         case 1: lam(0, i) = 1.0; break;
         case 2: lam(0, i) = 1.0; lam(2, i) = 1.0; break;
         case 3: lam(0, i) = 1.0; lam(3, i) = 1.0; lam(5, i) = 1.0; break;
         default: throw MachException("dim must be 1, 2 or 3.\n");
      }
      
      geom = mesh->GetElement(i)->GetGeometryType();
      eltransf = mesh->GetElementTransformation(i);
      eltransf->Transform(Geometries.GetCenter(geom), cent);

      diff = 0.0;
      for (int k = 0; k < dim; k++)
      {
         basis_center(k, i) = cent(k);
         if (i > 0)
         {
            diff(k) = basis_center(k, i) - basis_center(k, i-1);
         }
      }
      h += sqrt(diff.Norml2()); 
   }

   // for now we use an uniform span
   h /= (num_basis - 1);
   span = span_coef * h;
   cout << "RBF Space initilize complete, check data:\n";
   cout << "dim = " << dim << endl;
   cout << "num_basis = " << num_basis << endl;
   cout << "span = " << span << '\n';
   // cout << "Basis centers are:\n";
   // for (int j = 0; j < num_basis; j++)
   // {
   //    cout << j << ": ";
   //    for (int i = 0; i < dim; i++)
   //    {
   //       cout << basis_center(i, j) << ' ';
   //    }
   //    cout << endl;
   // }
   // cout << "lam are:\n";
   // for (int j = 0; j < num_basis; j++)
   // {
   //    cout << j << ": ";
   //    for (int i = 0; i < shape_param; i++)
   //    {
   //       cout << lam(i, j) << ' ';
   //    }
   //    cout << endl;
   //}
   BuildRBFProlongation();
}

void RBFSpace::BuildRBFProlongation() const
{
   // initialize the prolongation matrix
   cP = new mfem::SparseMatrix(GetVSize(), vdim * num_basis);
   cout << "cp initialize size " << cP->Height() << " x " <<  cP->Width() << '\n';
   // loop over each element to construct the prolongation operator
   Array<int> basis_selected;
   DenseMatrix basis_coord, inter_points, lam_selected, local_prolong;
   
   for (int i = 0; i < mesh->GetNE(); i++)
   {
      //cout << "Element " << i << ":\n";
      // 1. Get the basis patch
      SelectElementBasis(i, basis_selected, basis_coord, lam_selected);
      // cout << "Effective basis id: ";
      // basis_selected.Print(cout, basis_selected.Size());
      // cout << "Basis coords:\n";
      // basis_coord.Print(cout, basis_coord.Width());

      // 2. Get the elements' center matrix and quadrature points
      GetElementInterPoints(i, inter_points);
      // cout << "Quadrature points are: \n";
      // inter_points.Print(cout, inter_points.Width());
      

      // 3. Solve the basis coefficients
      buildRBFLSInterpolation(dim, basis_coord, inter_points,
                              lam_selected, local_prolong);
      //cout << "local prolongation built.\n";

      // 4. Assemble local_prolong back to global prolongation matrix
      AssembleProlongationMatrix(i, basis_selected, local_prolong);
      // cout<< "local prolongation matrix is:\n";
      // local_prolong.Print(cout, local_prolong.Width());
   }  
   cP->Finalize();
   cP_is_set = true;
   cout << "RBF prolongation matrix size: " << cP->Height() << " x " << cP->Width() << '\n';
   // ofstream cp_save("rfb_prolongation.txt");
   // cP->PrintMatlab(cp_save);
   // cp_save.close();
}

void RBFSpace::SelectElementBasis(const int id, 
                                  Array<int> &basis_selected,
                                  DenseMatrix &basis_coord,
                                  DenseMatrix &lam_selected) const
{
   basis_selected.LoseData();
   basis_coord.Clear();
   lam_selected.Clear();

   // Get the element center coordinate
   Vector cent(dim), diff(dim);
   int geom = mesh->GetElement(id)->GetGeometryType();
   ElementTransformation *eltransf = mesh->GetElementTransformation(id);
   eltransf ->Transform(Geometries.GetCenter(geom), cent);
   
   double dist;
   for (int j = 0; j < num_basis; j++)
   {
      // compute the distance
      for (int i = 0; i < dim; i++)
      {
         diff(i) = cent(i) - basis_center(i,j);
      }
      dist = diff.Norml2();
      if ( dist <= span)
      {
         basis_selected.Append(j);
      }
   }
   basis_coord.SetSize(dim, basis_selected.Size());
   lam_selected.SetSize(shape_param, basis_selected.Size());
   int i;
   for (int j = 0; j < basis_selected.Size(); j++)
   {
      for (i = 0; i < dim; i ++)
      {
         basis_coord(i,j) = basis_center(i, basis_selected[j]);
      }
      for (i = 0; i < shape_param; i++)
      {
         lam_selected(i,j) = lam(i, basis_selected[j]);
      }
   }

}

void RBFSpace::GetElementInterPoints(const int id, DenseMatrix &inter_points) const
{
   inter_points.Clear();

   // assume the mesh only contains only 1 type of element
   const Element* el = mesh->GetElement(id);
   const FiniteElement *fe = fec->FiniteElementForGeometry(el->GetGeometryType());
   const int num_dofs = fe->GetDof();

   inter_points.SetSize(dim, num_dofs);
   Vector quad_coord(dim);
   ElementTransformation *eltransf = mesh->GetElementTransformation(id);;
   for(int j = 0; j < num_dofs; j++)
   {
      eltransf->Transform(fe->GetNodes().IntPoint(j), quad_coord);
      for (int i = 0; i < dim; i++)
      {
         inter_points(i,j) = quad_coord(i);
      }
   }
}

void RBFSpace::AssembleProlongationMatrix(const int id,
                                          const Array<int> &basis_selected,
                                          const DenseMatrix &local_prolong) const
{
   const Element* el = mesh->GetElement(id);
   const FiniteElement *fe = fec->FiniteElementForGeometry(el->GetGeometryType());
   const int num_dofs = fe->GetDof();

   int local_num_basis = basis_selected.Size();
   Array<int> el_dofs;
   Array<int> col_index(local_num_basis);
   Array<int> row_index(num_dofs);

   GetElementVDofs(id, el_dofs);
   for(int e = 0; e < local_num_basis; e++)
   {
      col_index[e] = vdim * basis_selected[e];
   }

   for (int v = 0; v < vdim; v++)
   {
      el_dofs.GetSubArray(v * num_dofs, num_dofs, row_index);
      cP->SetSubMatrix(row_index, col_index, local_prolong, 1);
      row_index.LoseData();
      // elements id also need to be shift accordingly
      for (int e = 0; e < local_num_basis; e++)
      {
         col_index[e]++;
      }
   }
}

} // end of namespace mfem