#include <fstream>
#include <iostream>
#include "default_options.hpp"
#include "galer_diff.hpp"
#include "utils.hpp"

using namespace std;
using namespace mach;

namespace mfem
{

ParGDSpace::ParGDSpace(Mesh *m, ParMesh *pm, const FiniteElementCollection *fec,
                       int vdim, int ordering, int de, int p)
   : pr(p), full_mesh(m), SpaceType(pm,fec,vdim,ordering)
{

   degree = de;
   total_nel = full_mesh->GetNE();
   el_offset = GetParMesh()->GetGlobalElementNum(0);

   // re-cal necessary members from base classses
   gddofs = GetParMesh()->GetNE();
   tdof_offsets[0] = vdim * el_offset;
   tdof_offsets[1] = vdim *(el_offset + gddofs);


   // determine the the local prolongation matrix size

   col_start = vdim * el_offset;
   col_end = col_start + vdim * gddofs - 1;

   HYPRE_BigInt *offsets = GetDofOffsets();
   row_start = offsets[0];
   row_end = offsets[1]-1;


   local_tdof = vdim * GetParMesh()->GetNE();

   if (GetMyRank() == pr)
   {
      cout << "Constructint the parallel prolongation matrix:\n";
      cout << "vdim is " << vdim << endl;
      cout << "dof offsets are " << offsets[0] << ", " << offsets[1] << endl;
      cout << "tdof_offset is " << tdof_offsets[0] << ", " << tdof_offsets[1] << endl;
      cout << "row start and end are " << row_start << ", " << row_end << endl;
      cout << "col start and end are " << col_start << ", " << col_end << endl;
   }
   MPI_Barrier(GetComm());

   BuildProlongationOperator();

   HYPRE_BigInt ssize = GlobalTrueVSize();
   if (GetMyRank() == pr)
   {
      cout << "Global true Vsize is " << ssize << endl;
      cout << "HypreProlongation matrix size are " << P->Height() << " x " << P->Width() << endl;
   }
}

void ParGDSpace::GetNeighbourSet(int id, int req_n,
                                 mfem::Array<int> &nels)
{
   id = GetParMesh()->GetGlobalElementNum(id);
   // initialize the patch list
   nels.LoseData();
   nels.Append(id);
   // Creat the adjacent array and fill it with the first layer of adj
   // adjcant element list, candidates neighbors, candidates neighbors' adj
   Array<int> adj, cand, cand_adj, cand_next;
   full_mesh->ElementToElementTable().GetRow(id, adj);
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
         full_mesh->ElementToElementTable().GetRow(cand[i], cand_adj);
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


// an overload function of previous one (more doable?)
void ParGDSpace::BuildNeighbourMat(const mfem::Array<int> &elmt_id,
                                   mfem::DenseMatrix &mat_cent,
                                   mfem::DenseMatrix &mat_quad) const
{
   // assume the mesh only contains only 1 type of element
   const Element* el = full_mesh->GetElement(0);
   const FiniteElement *fe = fec->FiniteElementForGeometry(el->GetGeometryType());
   const int num_dofs = fe->GetDof();
   const int dim = full_mesh->Dimension();

   // resize the DenseMatrices and clean the data
   int num_el = elmt_id.Size();
   mat_cent.Clear(); 
   mat_cent.SetSize(dim, num_el);

   Vector cent_coord(dim);
   full_mesh->GetElementCenter(elmt_id[0], cent_coord);
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
      full_mesh->GetElementCenter(elmt_id[j], cent_coord);
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
   ElementTransformation *eltransf = full_mesh->GetElementTransformation(elmt_id[0]);;
   for(int i = 0; i < num_dofs; i++)
   {
      eltransf->Transform(fe->GetNodes().IntPoint(i), quad_coord);
      for(int j = 0; j < dim; j++)
      {
         mat_quad(j,i) = quad_coord(j);
      }
   }
}

void ParGDSpace::BuildProlongationOperator()
{

   HYPRE_IJMatrixCreate(GetComm(),row_start,row_end,col_start,col_end,&ij_matrix);
   //HYPRE_IJMatrixCreate(GetComm(),);
   HYPRE_IJMatrixSetObjectType(ij_matrix,HYPRE_PARCSR);
   HYPRE_IJMatrixInitialize(ij_matrix);

   // assume the mesh only contains only 1 type of element
   const Element* el = full_mesh->GetElement(0);
   const FiniteElement *fe = fec->FiniteElementForGeometry(el->GetGeometryType());
   const int num_dofs = fe->GetDof();
   const int dim = full_mesh->Dimension();

   // cP = new mfem::SparseMatrix(full_fespace->GetVSize(), vdim * total_nel);
   // determine the minimum # of element in each patch
   int nelmt;
   switch(full_mesh->Dimension())
   {
      case 1: nelmt = degree + 1; break;
      case 2: nelmt = (degree+1) * (degree+2) / 2; break;
      case 3: throw MachException("Not implemeneted yet.\n"); break;
      default: throw MachException("dim must be 1, 2 or 3.\n");
   }

   // cout << "cp size is " << cP->Height() << " x " << cP->Width() << '\n';
   // vector that contains element id (resize to zero )
   mfem::Array<int> elmt_id;
   mfem::DenseMatrix cent_mat, quad_mat, local_mat;
   for (int i = 0; i < GetParMesh()->GetNE(); i++)
   {
      // 1. Get element id in patch
      GetNeighbourSet(i, nelmt, elmt_id);
      // if (GetMyRank() == pr)
      // {
      //    cout << "id(s) in patch " << i << ": ";
      //    elmt_id.Print(cout, elmt_id.Size());
      // }


      // 2. build the quadrature and barycenter coordinate matrices
      BuildNeighbourMat(elmt_id, cent_mat, quad_mat);
      // if (GetMyRank() == pr)
      // {
      //    cout << "The element center matrix:\n";
      //    cent_mat.Print(cout, cent_mat.Width());
      // }
      // cout << "Quadrature points id matrix:\n";
      // quad_mat.Print(cout, quad_mat.Width());
      // cout << endl;

      // 3. buil the local reconstruction matrix
      buildLSInterpolation(dim, degree, cent_mat, quad_mat, local_mat);
      // cout << "Local reconstruction matrix R:\n";
      // local_mat.Print(cout, local_mat.Width());
      // cout << endl;
      // cout << endl;
   
      // 4. assemble them back to prolongation matrix
      AssembleProlongationMatrix(elmt_id, local_mat);
   }
   HYPRE_IJMatrixAssemble(ij_matrix);
   HYPRE_IJMatrixGetObject(ij_matrix, (void**)&prolong);
   P = new HypreParMatrix((hypre_ParCSRMatrix*)(prolong), true);
   P->Print("prolong");
   // Vector diag(local_tdof);
   // diag = 1.0;
   // R = new SparseMatrix(diag);
   // if (pr == GetMyRank())
   // {
   //    cout << "R size is " << R->Height() << " x " << R->Width() << endl;
   // }
}

void ParGDSpace::AssembleProlongationMatrix(const mfem::Array<int> &id,
                                            const DenseMatrix &local_mat) const
{
   // element id coresponds to the column indices
   // dofs id coresponds to the row indices
   // the local reconstruction matrix needs to be assembled `vdim` times
   // assume the mesh only contains only 1 type of element
   int nel = id.Size();
   const int main_id_global = id[0];
   const int main_id_local  = GetParMesh()->GetLocalElementNum(main_id_global); 
   const Element* el = full_mesh->GetElement(0);
   const FiniteElement *fe = fec->FiniteElementForGeometry(el->GetGeometryType());
   const int num_dofs = fe->GetDof();
   MFEM_VERIFY(num_dofs == local_mat.Height(),"matrix height doesn't match # of dof");
   
   Array<int> el_dofs;
   int dof_offset = GetMyDofOffset();
   GetElementVDofs(main_id_local, el_dofs);
   for (int i = 0; i < el_dofs.Size(); i++)
   {
      el_dofs[i] += dof_offset;
   }

   int j, v, e;
   int row_index;
   Array<int> col_index(nel);
   Vector single_row;

   for (v = 0; v < vdim; v++)
   {
      for (e = 0; e < nel; e++)
      {
         col_index[e] = vdim * id[e] + v;
      }

      for (j = 0; j < num_dofs; j++)
      {
         local_mat.GetRow(j, single_row);
         row_index = el_dofs[v*num_dofs+j];
         HYPRE_IJMatrixSetValues(ij_matrix,1,&nel, &row_index, col_index.GetData(),
                                 single_row.GetData());
      }
   }
}

const Operator *ParGDSpace::GetProlongationMatrix() const
{
   return P;
}

} // end of namespace