
#include "galer_diff.hpp"
#include <fstream>
#include <iostream>
#include "sbp_fe.hpp"
using namespace std;
using namespace mach;
using namespace mfem;

namespace mfem
{

GalerkinDifference::GalerkinDifference(Mesh *pm, const FiniteElementCollection *f,
   int vdim, int ordering, int de)
   : SpaceType(pm, f, vdim, ordering)
{
   degree = de;
   nEle = mesh->GetNE();
   dim = mesh->Dimension();
   fec = f;
   BuildGDProlongation();
}

// an overload function of previous one (more doable?)
void GalerkinDifference::BuildNeighbourMat(const mfem::Array<int> &elmt_id,
                                           mfem::DenseMatrix &mat_cent,
                                           mfem::DenseMatrix &mat_quad) const
{
   // assume the mesh only contains only 1 type of element
   const Element* el = mesh->GetElement(0);
   const FiniteElement *fe = fec->FiniteElementForGeometry(el->GetGeometryType());
   const int num_dofs = fe->GetDof();

   // resize the DenseMatrices and clean the data
   int num_el = elmt_id.Size();
   mat_cent.Clear(); 
   mat_cent.SetSize(dim, num_el);

   Vector cent_coord(dim);
   GetElementCenter(elmt_id[0], cent_coord);
   double left_threshold = 0.25;
   double right_threshold = 0.75;
   double top_threshold = 0.75;
   double bot_threshold = 0.25;
   bool left = false, right = false, top = false, bot = false;

   if (cent_coord(0) > right_threshold) {right = true;}
   if (cent_coord(0) < left_threshold) {left = true;}
   if (cent_coord(1) < bot_threshold) {bot = true;}
   if (cent_coord(1) > top_threshold) {top = true;}

   for(int j = 0; j < num_el; j++)
   {
      // Get and store the element center
      GetElementCenter(elmt_id[j], cent_coord);

      if (right)
      {
         if (cent_coord(0)+1.0 < 1.5 )
         {
            cent_coord(0) = cent_coord(0) + 1.0;
         }
      }

      if (left)
      {
         if (cent_coord(0)-1.0 > -0.5)
         {
            cent_coord(0) = cent_coord(0) - 1.0;
         }
      }

      if (top)
      {
         if (cent_coord(1)+1.0 < 1.5)
         {
            cent_coord(1) = cent_coord(1) + 1.0;
         }
      }

      if (bot)
      {
         if (cent_coord(1)-1.0 > -0.5)
         {
            cent_coord(1) = cent_coord(1) - 1.0;
         }
      }
      

      for(int i = 0; i < dim; i++)
      {
         mat_cent(i,j) = cent_coord(i);
      }
   }

   Vector quad_coord(dim);
   mat_quad.Clear();
   mat_quad.SetSize(dim, num_dofs);
   ElementTransformation *eltransf = mesh->GetElementTransformation(elmt_id[0]);;
   for(int i = 0; i < num_dofs; i++)
   {
      eltransf->Transform(fe->GetNodes().IntPoint(i), quad_coord);
      for(int j = 0; j < dim; j++)
      {
         mat_quad(j,i) = quad_coord(j);
      }
   }
}

void GalerkinDifference::GetNeighbourSet(int id, int req_n,
                                    mfem::Array<int> &nels) const
{
   // using mfem mesh object to construct the element patch
   // initialize the patch list
   nels.LoseData();
   nels.Append(id);
   // Creat the adjacent array and fill it with the first layer of adj
   // adjcant element list, candidates neighbors, candidates neighbors' adj
   Array<int> adj, cand, cand_adj, cand_next;
   mesh->ElementToElementTable().GetRow(id, adj);
   cand.Append(adj);
   while(nels.Size() < req_n)
   {
      for(int i = 0; i < adj.Size(); i++)
      {
         if (-1 == nels.Find(adj[i]))
         {
            nels.Append(adj[i]); 
         }
      }
      adj.LoseData();
      for (int i = 0; i < cand.Size(); i++)
      {
         //cout << "deal with cand " << cand[i];
         mesh->ElementToElementTable().GetRow(cand[i], cand_adj);
         //cout << "'s adj are ";
         //cand_adj.Print(cout, cand_adj.Size());
         for(int j = 0; j < cand_adj.Size(); j++)
         {
            if (-1 == nels.Find(cand_adj[j]))
            {
               //cout << cand_adj[j] << " is not found in nels. add to adj and cand_next.\n";
               adj.Append(cand_adj[j]);
               cand_next.Append(cand_adj[j]);
            }
         }
         cand_adj.LoseData();
      }
      cand.LoseData();
      cand = cand_next;
      cand_next.LoseData();
   }
}

void GalerkinDifference::GetElementCenter(int id, mfem::Vector &cent) const
{
   cent.SetSize(mesh->Dimension());
   int geom = mesh->GetElement(id)->GetGeometryType();
   ElementTransformation *eltransf = mesh->GetElementTransformation(id);
   eltransf->Transform(Geometries.GetCenter(geom), cent);
}

void GalerkinDifference::BuildGDProlongation() const
{
   // assume the mesh only contains only 1 type of element
   const Element* el = mesh->GetElement(0);
   const FiniteElement *fe = fec->FiniteElementForGeometry(el->GetGeometryType());
   const int num_dofs = fe->GetDof();
   // allocate the space for the prolongation matrix
   // this step should be done in the constructor (probably)
   // should it be GetTrueVSize() ? or GetVSize()?
   // need a new method that directly construct a CSR format sparsematrix ？
   cP = new mfem::SparseMatrix(GetVSize(), vdim * nEle);
   // determine the minimum # of element in each patch
   int nelmt;
   switch(dim)
   {
      case 1: nelmt = degree + 1; break;
      case 2: nelmt = (degree+1) * (degree+2) / 2; break;
      case 3: throw MachException("Not implemeneted yet.\n"); break;
      default: throw MachException("dim must be 1, 2 or 3.\n");
   }
   // cout << "Number of required element: " << nelmt << '\n';
   // loop over all the element:
   // 1. build the patch for each element,
   // 2. construct the local reconstruction operator
   // 3. assemble local reconstruction operator
   
   // vector that contains element id (resize to zero )
   mfem::Array<int> elmt_id;
   mfem::DenseMatrix cent_mat, quad_mat, local_mat;
   cout << "The size of the prolongation matrix is " << cP->Height() << " x " << cP->Width() << '\n';
   //int degree_actual;
   for (int i = 0; i < nEle; i++)
   {
      GetNeighbourSet(i, nelmt, elmt_id);
      // cout << "id(s) in patch " << i << ": ";
      // elmt_id.Print(cout, elmt_id.Size());
      
      // 2. build the quadrature and barycenter coordinate matrices
      BuildNeighbourMat(elmt_id, cent_mat, quad_mat);
      // cout << "The element center matrix:\n";
      // cent_mat.Print(cout, cent_mat.Width());
      // cout << endl;
      // cout << "Quadrature points id matrix:\n";
      // quad_mat.Print(cout, quad_mat.Width());
      // cout << endl;

      // 3. buil the loacl reconstruction matrix
      buildLSInterpolation(dim, degree, cent_mat, quad_mat, local_mat);
      // cout << "Local reconstruction matrix R:\n";
      // local_mat.Print(cout, local_mat.Width());
      // cout << endl;
      // cout << endl;

      // 4. assemble them back to prolongation matrix
      AssembleProlongationMatrix(elmt_id, local_mat);
   }
   cP->Finalize();
   cP_is_set = true;
   cout << "Check cP size: " << cP->Height() << " x " << cP->Width() << '\n';
   ofstream cp_save("cP.txt");
   cP->PrintMatlab(cp_save);
   cp_save.close();
}

void GalerkinDifference::AssembleProlongationMatrix(const mfem::Array<int> &id,
                                            const DenseMatrix &local_mat) const
{
   // element id coresponds to the column indices
   // dofs id coresponds to the row indices
   // the local reconstruction matrix needs to be assembled `vdim` times
   // assume the mesh only contains only 1 type of element
   const Element* el = mesh->GetElement(0);
   const FiniteElement *fe = fec->FiniteElementForGeometry(el->GetGeometryType());
   const int num_dofs = fe->GetDof();

   int nel = id.Size();
   Array<int> el_dofs;
   Array<int> col_index(nel);
   Array<int> row_index(num_dofs);

   int el_id = id[0];
   GetElementVDofs(el_id, el_dofs);
   // cout << "element vdofs is: ";
   // el_dofs.Print(cout, el_dofs.Size());
   for(int e = 0; e < nel; e++)
   {
      col_index[e] = vdim * id[e];
   }

   for (int v = 0; v < vdim; v++)
   {
      el_dofs.GetSubArray(v * num_dofs, num_dofs, row_index);
      cP->SetSubMatrix(row_index, col_index, local_mat, 1);
      row_index.LoseData();
      // elements id also need to be shift accordingly
      for (int e = 0; e < nel; e++)
      {
         col_index[e]++;
      }
   }
}

} // namespace mfem