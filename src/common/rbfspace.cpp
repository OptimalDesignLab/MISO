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
   lam.SetSize(num_basis);
   basis_center.SetSize(num_basis);
   a_test.reset(new Array<Vector>(num_basis));
   cout << "a_test size is " << a_test->Size();

   int shape_param;
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

   // initialize the basis centers and span
   int geom;
   ElementTransformation *eltransf;
   Vector diff;
   double h = 0.0;
   for (int i = 0; i < num_basis; i++)
   {
      lam[i].SetSize(shape_param);
      lam[i] = 0.0;
      switch (dim)
      {
         case 1: lam[i](0) = 1.0; break;
         case 2: lam[i](0) = 1.0; lam[i](2) = 1.0; break;
         case 3: lam[i](0) = 1.0; lam[i](3) = 1.0; lam[i](5) = 1.0; break;
         default: throw MachException("dim must be 1, 2 or 3.\n");
      }
      
      geom = mesh->GetElement(i)->GetGeometryType();
      eltransf = mesh->GetElementTransformation(i);
      eltransf->Transform(Geometries.GetCenter(geom), basis_center[i]);

      if (i > 0)
      {
         diff = basis_center[i];
         diff -= basis_center[i-1];
         h += sqrt(diff.Norml2());
      }
   }

   // for now we use an uniform span
   h /= (num_basis - 1);
   span = span_coef * h;
   cout << "RBF Space initilize complete, check data:\n";
   cout << "dim = " << dim << endl;
   cout << "num_basis = " << num_basis << endl;
   cout << "span = " << span << '\n'; 
   cout << "basis_center:\n";
   for (int i = 0; i < num_basis; i++)
   {
      cout << i << ": ";
      basis_center[i].Print(cout, dim);
   }
   cout << "lams are:\n";
   for (int i = 0; i < num_basis; i++)
   {
      cout << i << ": ";
      lam[i].Print(cout, shape_param);
   }
   

   BuildRBFProlongation();
}

void RBFSpace::BuildRBFProlongation() const
{
   // initialize the prolongation matrix
   cP = new mfem::SparseMatrix(GetVSize(), vdim * num_basis);
   cout << "cp initialize size " << cP->Height() << " x " <<  cP->Width() << '\n';
   // loop over each element to construct the prolongation operator
   Array<int> basis_selected;
   Array<Vector> basis_coord, inter_points;
   Array<Vector> lam_selected;
   DenseMatrix local_prolong;
   
   for (int i = 0; i < mesh->GetNE(); i++)
   {
      // 1. Get the basis patch
      SelectElementBasis(i, basis_selected, basis_coord, lam_selected);
      cout << "Element " << i << ":\n";
      cout << "Effective basis id: ";
      basis_selected.Print(cout, basis_selected.Size());
      cout << "Selected lambda: ";
      //lam_selected.Print(cout, lam_selected.Size());

      // 2. Get the elements' center matrix and quadrature points
      GetElementInterPoints(i, inter_points);
      

      // 3. Solve the basis coefficients
      buildRBFLSInterpolation(dim, basis_coord, inter_points, lam, local_prolong);

      // 4. Assemble local_prolong back to global prolongation matrix
      AssembleProlongationMatrix(i, basis_selected, local_prolong);
   }
   cP->Finalize();
   cP_is_set = true;
   cout << "RBF prolongation matrix size: " << cP->Height() << " x " << cP->Width() << '\n';
   ofstream cp_save("rfb_prolongation.txt");
   cP->PrintMatlab(cp_save);
   cp_save.close();
}

void RBFSpace::SelectElementBasis(const int id, Array<int> &basis_selected,
                                  Array<Vector> &basis_coord,
                                  Array<Vector> &lam_selected) const
{
   cout << "In selectelement basis: \n";
   basis_selected.LoseData();
   basis_coord.LoseData();
   lam_selected.LoseData();
   cout << "data losed, size is " << basis_selected.Size() << basis_coord.Size() <<  '\n';

   // Get the element center coordinate
   Vector cent(dim), diff(dim);
   int geom = mesh->GetElement(id)->GetGeometryType();
   ElementTransformation *eltransf = mesh->GetElementTransformation(id);
   eltransf ->Transform(Geometries.GetCenter(geom), cent);
   
   double dist;
   for (int j = 0; j < num_basis; j++)
   {
      // compute the distance
      diff = cent;
      diff -= basis_center[j];
      dist = diff.Norml2();
      cout << "distance is " << dist << '\n';
      if ( dist <= span)
      {
         basis_selected.Append(j);
         cout << "basis_selected size is " << basis_selected.Size() << endl;
         basis_coord.Append(basis_center[j]);
         cout << "basis_coord size is " << basis_coord.Size() << endl;
         // lam_selected.Append(lam[j]);
         // cout << "lam_selected size is " << lam_selected.Size() << endl;
         // cout << "append lam.\n";
      }
   }
   cout << "size of variables: " << basis_selected.Size() << ' '
        << basis_coord.Size() << endl;
}

void RBFSpace::GetElementInterPoints(const int id, Array<Vector> &inter_points) const
{
   inter_points.LoseData();

   // assume the mesh only contains only 1 type of element
   const Element* el = mesh->GetElement(id);
   const FiniteElement *fe = fec->FiniteElementForGeometry(el->GetGeometryType());
   const int num_dofs = fe->GetDof();

   Vector quad_coord(dim);
   ElementTransformation *eltransf = mesh->GetElementTransformation(id);;
   for(int i = 0; i < num_dofs; i++)
   {
      eltransf->Transform(fe->GetNodes().IntPoint(i), quad_coord);
      inter_points.Append(quad_coord);
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