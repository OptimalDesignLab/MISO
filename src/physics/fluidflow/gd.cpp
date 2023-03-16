
#include "gd.hpp"
#include <fstream>
#include <iostream>
#include <numeric>
#include <algorithm>
#include <vector>
#include "pcentgridfunc.hpp"
using namespace std;
using namespace mfem;
extern "C" void dgecon_(char *,
                        int *,
                        double *,
                        int *,
                        double *,
                        double *,
                        double *,
                        int *,
                        int *);

extern "C" void dgelss_(int *,
                        int *,
                        int *,
                        double *,
                        int *,
                        double *,
                        int *,
                        double *,
                        double *,
                        int *,
                        double *,
                        int *,
                        int *);
extern "C" void dgels_(char *,
                       int *,
                       int *,
                       int *,
                       double *,
                       int *,
                       double *,
                       int *,
                       double *,
                       int *,
                       int *);

extern "C" void dgelsy_(int *,
                        int *,
                        int *,
                        double *,
                        int *,
                        double *,
                        int *,
                        int *,
                        double *,
                        int *,
                        double *,
                        int *,
                        int *);

// namespace mfem
// {

// }

ParGalerkinDifference::ParGalerkinDifference(
    mach::MeshType *pm,
    const FiniteElementCollection *f,
    std::vector<bool> _embeddedElements,
    int vdim,
    int ordering,
    int de,
    MPI_Comm _comm)
 : ParFiniteElementSpace(pm, f, vdim, ordering)
{
   degree = de;
   nEle = pm->GetNE();
   dim = pm->Dimension();
   fec = f;
   comm = _comm;
   embeddedElements = _embeddedElements;

   el_offset = GetParMesh()->GetGlobalElementNum(0);
   cout << "el_offset " << el_offset << endl;
   // re-cal necessary members from base classses
   gddofs = GetParMesh()->GetNE();
   cout << "gddofs " << gddofs << endl;
   tdof_offsets = new HYPRE_Int[2];
   tdof_offsets[0] = vdim * el_offset;
   tdof_offsets[1] = vdim * (el_offset + gddofs);
   cout << "tdof_offsets done" << endl;
   // determine the the local prolongation matrix size

   col_start = vdim * el_offset;
   col_end = col_start + vdim * gddofs - 1;
   cout << "b4 GetDofOffsets()" << endl;
   HYPRE_BigInt *offsets = GetDofOffsets();
   cout << "offsets done" << endl;
   row_start = offsets[0];
   row_end = offsets[1] - 1;

   local_tdof = vdim * GetParMesh()->GetNE();

   cout << "Constructing the parallel prolongation matrix:\n";
   cout << "vdim is " << vdim << endl;
   cout << "dof offsets are " << offsets[0] << ", " << offsets[1] << endl;
   cout << "tdof_offset is " << tdof_offsets[0] << ", " << tdof_offsets[1]
        << endl;
   cout << "row start and end are " << row_start << ", " << row_end << endl;
   cout << "col start and end are " << col_start << ", " << col_end << endl;
   MPI_Barrier(GetComm());

   BuildGDProlongation();

   HYPRE_BigInt ssize = GlobalTrueVSize();

   cout << "Global true Vsize is " << ssize << endl;
   cout << "HypreProlongation matrix size are " << P->Height() << " x "
        << P->Width() << endl;
}
void ParGalerkinDifference::Build_Dof_TrueDof_Matrix()
{
   if (!P)
   {
      BuildGDProlongation();
   }
#if 0
      HYPRE_Int row_size = cP->Height();
      HYPRE_Int col_size = cP->Width();
      cout << " row_size " << row_size << endl;
      mat_row_idx = new HYPRE_Int[2];
      mat_col_idx = new HYPRE_Int[2];
      mat_row_idx[0] = 0;
      mat_row_idx[1] = cP->Height();
      mat_col_idx[0] = 0;
      mat_col_idx[1] = cP->Width();
      cout << "creating P matrix " << endl;
      P = new HypreParMatrix(
          comm, row_size, col_size, mat_row_idx, mat_col_idx, cP);
      // P = new HypreParMatrix(
      //     comm, mat_row_idx, mat_col_idx, cP);
      cout << "P size " << P->Height() << " x " << P->Width() << endl;
      SparseMatrix Pdiag;
      P->GetDiag(Pdiag);
      R = Transpose(Pdiag);
#endif
}

HypreParMatrix *ParGalerkinDifference::Dof_TrueDof_Matrix() const
{
   // if (!P)
   // {
   //    cout << "P not there " << endl;
   //    Build_Dof_TrueDof_Matrix();
   // }
   return P;
}

// an overload function of previous one (more doable?)
void ParGalerkinDifference::BuildNeighbourMat(const mfem::Array<int> &elmt_id,
                                              mfem::DenseMatrix &mat_quad) const
{
   // resize the DenseMatrices and clean the data
   int num_el = elmt_id.Size();
   // cout << "num_el " << num_el << endl;
   // mat_cent.Clear();
   // mat_cent.SetSize(dim, num_el);

   // assume the mesh only contains only 1 type of element
   const Element *el = mesh->GetElement(0);
   const FiniteElement *fe =
       fec->FiniteElementForGeometry(el->GetGeometryType());
   const int num_dofs = fe->GetDof();
   // vectors that hold coordinates of quadrature points
   // used for duplication tests
   vector<double> quad_data;
   Vector quad_coord(dim);  // used to store quadrature coordinate temperally
   ElementTransformation *eltransf;
   eltransf = mesh->GetElementTransformation(elmt_id[0]);

   for (int k = 0; k < num_dofs; k++)
   {
      eltransf->Transform(fe->GetNodes().IntPoint(k), quad_coord);
      for (int di = 0; di < dim; di++)
      {
         quad_data.push_back(quad_coord(di));
      }
   }

   // for (int j = 0; j < num_el; j++)
   // {
   //    // Get and store the element center
   //    mfem::Vector cent_coord(dim);
   //    GetElementCenter(elmt_id[j], cent_coord);
   //    for (int i = 0; i < dim; i++)
   //    {
   //       mat_cent(i, j) = cent_coord(i);
   //    }
   // }

   // reset the quadrature point matrix
   mat_quad.Clear();
   int num_col = quad_data.size() / dim;
   mat_quad.SetSize(dim, num_col);
   for (int i = 0; i < num_col; i++)
   {
      for (int j = 0; j < dim; j++)
      {
         mat_quad(j, i) = quad_data[i * dim + j];
      }
   }
}
void ParGalerkinDifference::InitializeNeighbors(int id,
                                                int req_n,
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
   // cout << "List is initialized as: ";
   // nels.Print(cout, nels.Size());
   // cout << "Initial candidates: ";
   // cand.Print(cout, cand.Size());
   // cout << "req_n " << req_n << endl;
   while (nels.Size() < req_n)
   {
      for (int i = 0; i < adj.Size(); i++)
      {
         if (-1 == nels.Find(adj[i]))
         {
            if (embeddedElements.at(adj[i]) == false)
            {
               nels.Append(adj[i]);
            }
         }
      }
      // cout << "List now is: ";
      // nels.Print(cout, nels.Size());
      adj.LoseData();
      for (int i = 0; i < cand.Size(); i++)
      {
         if (embeddedElements.at(cand[i]) == false)
         {
            // cout << "deal with cand " << cand[i];
            mesh->ElementToElementTable().GetRow(cand[i], cand_adj);
            // cout << "'s adj are ";
            // cand_adj.Print(cout, cand_adj.Size());
            for (int j = 0; j < cand_adj.Size(); j++)
            {
               if (-1 == nels.Find(cand_adj[j]))
               {
                  // cout << cand_adj[j] << " is not found in nels. add to adj
                  // and cand_next.\n";
                  adj.Append(cand_adj[j]);
                  cand_next.Append(cand_adj[j]);
               }
            }
            cand_adj.LoseData();
         }
      }
      cand.LoseData();
      cand = cand_next;
      // cout << "cand copy from next: ";
      // cand.Print(cout, cand.Size());
      cand_next.LoseData();
   }
}
void ParGalerkinDifference::SortNeighbors(int id,
                                          int req_n,
                                          const Array<int> &els_id,
                                          Array<int> &nels) const
{
   std::vector<size_t> sortedEBDistRank;
   nels.LoseData();
   std::map<int, double> neighborDist;
   Element *el = mesh->GetElement(id);
   mfem::Array<int> v;
   el->GetVertices(v);
   Vector center(dim), refCent(dim);
   GetElementCenter(id, center);
   // cout << "element center: " << endl;
   // center.Print();
   ElementTransformation *eltransf = mesh->GetElementTransformation(id);
   // cout << "element size: " << mesh->GetElementSize(id) << endl;
   // double el_size = mesh->GetElementSize(id);
   // cout << "coords " << endl;
   // for (int i = 0; i < v.Size(); ++i)
   // {
   //    double *coord = mesh->GetVertex(v[i]);
   //    // cout << coord[0] << " , " << coord[1] << endl;
   //    if (coord[0] < center(0) && coord[1] < center(1))
   //    {
   //       refCent(0) = 0.5 * (coord[0] + center(0));
   //       refCent(1) = 0.5 * (coord[1] + center(1));
   //    }
   // }
   // refCent(0) = center(0) - 0.25*el_size;
   // refCent(1) = center(1) - 0.25*el_size;

   /// required for conforming mesh case
   refCent(0) = center(0);
   refCent(1) = center(1);
   // if (id == 3)
   // {
   // cout << "ref center :" << endl;
   // refCent.Print();
   // }
   vector<double> elementBasisDist;
   elementBasisDist.clear();
   sortedEBDistRank.clear();
   double dist;
   // loop over all basis
   for (int j = 0; j < els_id.Size(); ++j)
   {
      Vector elemCenter(dim);
      GetElementCenter(els_id[j], elemCenter);
      elemCenter -= refCent;
      dist = elemCenter.Norml2();
      elementBasisDist.push_back(dist);
   }
   // build element/basis stencil based on distance
   sortedEBDistRank = sort_indexes(elementBasisDist);
   for (int k = 0; k < els_id.Size(); k++)
   {
      int bid = sortedEBDistRank[k];
      int sid = els_id[bid];
      nels.Append(sid);
   }

   // check if the stencil selection is valid
   // for (i = 0; i < numBasis; i++)
   // {
   //    if (selectedBasis.empty())
   //    {
   //       cout << "Basis " << i << " is not selected.\n";
   //       throw MachException("Basis center is not selected.");
   //    }
   // }
}

vector<size_t> ParGalerkinDifference::sort_indexes(
    const vector<double> &v) const
{
   // initialize original index locations
   vector<size_t> idx(v.size());
   iota(idx.begin(), idx.end(), 0);

   // sort indexes based on comparing values in v
   // using std::stable_sort instead of std::sort
   // to avoid unnecessary index re-orderings
   // when v contains elements of equal values
   stable_sort(idx.begin(),
               idx.end(),
               [&v](size_t i1, size_t i2) { return v[i1] < v[i2]; });

   return idx;
}

void ParGalerkinDifference::ConstructStencil(int id,
                                             int req_n,
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
   // cout << "List is initialized as: ";
   // nels.Print(cout, nels.Size());
   // cout << "Initial candidates: ";
   // cand.Print(cout, cand.Size());
   // cout << "req_n " << req_n << endl;
   while (nels.Size() < req_n)
   {
      for (int i = 0; i < adj.Size(); i++)
      {
         if (-1 == nels.Find(adj[i]))
         {
            if (embeddedElements.at(adj[i]) == false)
            {
               nels.Append(adj[i]);
            }
         }
      }
      // cout << "List now is: ";
      // nels.Print(cout, nels.Size());
      adj.LoseData();
      for (int i = 0; i < cand.Size(); i++)
      {
         if (embeddedElements.at(cand[i]) == false)
         {
            // cout << "deal with cand " << cand[i];
            mesh->ElementToElementTable().GetRow(cand[i], cand_adj);
            // cout << "'s adj are ";
            // cand_adj.Print(cout, cand_adj.Size());
            for (int j = 0; j < cand_adj.Size(); j++)
            {
               if (-1 == nels.Find(cand_adj[j]))
               {
                  // cout << cand_adj[j] << " is not found in nels. add to adj
                  // and cand_next.\n";
                  adj.Append(cand_adj[j]);
                  cand_next.Append(cand_adj[j]);
               }
            }
            cand_adj.LoseData();
         }
      }
      cand.LoseData();
      cand = cand_next;
      // cout << "cand copy from next: ";
      // cand.Print(cout, cand.Size());
      cand_next.LoseData();
   }
}

void ParGalerkinDifference::GetElementCenter(int id, mfem::Vector &cent) const
{
   cent.SetSize(mesh->Dimension());
   int geom = mesh->GetElement(id)->GetGeometryType();
   ElementTransformation *eltransf = mesh->GetElementTransformation(id);
   eltransf->Transform(Geometries.GetCenter(geom), cent);
}

void ParGalerkinDifference::BuildGDProlongation()
{
   HYPRE_IJMatrixCreate(
       comm, row_start, row_end, col_start, col_end, &ij_matrix);
   // HYPRE_IJMatrixCreate(GetComm(),);
   HYPRE_IJMatrixSetObjectType(ij_matrix, HYPRE_PARCSR);
   HYPRE_IJMatrixInitialize(ij_matrix);
   // assume the mesh only contains only 1 type of element
   const Element *el = mesh->GetElement(0);
   const FiniteElement *fe =
       fec->FiniteElementForGeometry(el->GetGeometryType());
   const int num_dofs = fe->GetDof();
   // allocate the space for the prolongation matrix
   // this step should be done in the constructor (probably)
   // should it be GetTrueVSize() ? or GetVSize()?
   // need a new method that directly construct a CSR format sparsematrix ？
   // cP = new mfem::SparseMatrix(GetVSize(), vdim * nEle);
   // determine the minimum # of element in each patch
   int nreq, nreq_init;
   switch (dim)
   {
   case 1:
      nreq = degree + 1;
      break;
   case 2:
      nreq = (degree + 1) * (degree + 2) / 2;
      nreq_init = (degree + 2) * (degree + 3) / 2;
      // nreq = nreq + 1; // experimenting
      // nreq_init = nreq;  /// old stencil
      break;
   case 3:
      cout << "Not implemeneted yet.\n" << endl;
      break;
   default:
      cout << "dim must be 1, 2 or 3.\n" << endl;
   }
   // cout << "Number of required element: " << nelmt << '\n';
   // loop over all the element:
   // 1. build the patch for each element,
   // 2. construct the local reconstruction operator
   // 3. assemble local reconstruction operator

   // vector that contains element id (resize to zero )
   mfem::Array<int> elmt_id, nels, stencil_elid;
   mfem::DenseMatrix cent_mat, quad_mat, local_mat, V;
   // cout << "The size of the prolongation matrix is " << cP->Height() << " x "
   //      << cP->Width() << '\n';
   // int degree_actual;
   for (int i = 0; i < nEle; i++)
   {
      if (embeddedElements.at(i) == true)
      {
         // cout << "embedded element: " << i << endl;
         elmt_id.LoseData();
         elmt_id.Append(i);

         local_mat.SetSize(num_dofs, 1);

         for (int k = 0; k < num_dofs; ++k)
         {
            local_mat(k, 0) = 0.0;
         }

         AssembleProlongationMatrix(elmt_id, local_mat);
      }
      else
      {
         ConstructStencil(i, nreq_init, nels);
         SortNeighbors(i, nreq, nels, elmt_id);
         // cout << "#elements in stencil first " << nels.Size() << endl;
         // cout << "Elements id(s) in initial patch " << i << ": " << endl;
         // nels.Print(cout, nels.Size());
         // cout << " =================================================== "
         //      << endl;

         // cout << "building vandermonde for element: " << i << endl;
         // buildVandermondeMat(dim, nreq, elmt_id, stencil_elid, cent_mat, V);
         buildVandermondeMat(dim,
                             nreq,
                             nels,
                             stencil_elid,
                             cent_mat,
                             V);  // old stencil approach
         if (i == 365 || i == 716 || i == 429)
         {
            cout << " =================================================== "
                 << endl;
            cout << "#elements in final stencil " << stencil_elid.Size()
                 << endl;
            cout << "Elements id(s) in stencil of " << i << ": " << endl;
            stencil_elid.Print(cout, stencil_elid.Size());
            cout << " =================================================== "
                 << endl;
         }

         // 2. build the quadrature and barycenter coordinate matrices
         BuildNeighbourMat(stencil_elid, quad_mat);
         // cout << "neighbour mat is done " << endl;
         //  3. buil the loacl reconstruction matrix
         buildLSInterpolation(i, dim, degree, V, cent_mat, quad_mat, local_mat);
         // cout << "build LS interpolation " << endl;
         //  4. assemble them back to prolongation matrix
         AssembleProlongationMatrix(stencil_elid, local_mat);
         // cout << "assemble prolongation done " << endl;
      }
   }
   HYPRE_IJMatrixAssemble(ij_matrix);
   HYPRE_IJMatrixGetObject(ij_matrix, (void **)&prolong);
   P = new HypreParMatrix((hypre_ParCSRMatrix *)(prolong), true);
   P->Print("prolong");
   Vector diag(local_tdof);
   diag = 1.0;
   R = new SparseMatrix(diag);
   if (pr == GetMyRank())
   {
      cout << "R size is " << R->Height() << " x " << R->Width() << endl;
   }
/// serial case
#if 0
   cP->Finalize();
   cP_is_set = true;
   cout << "Check cP size: " << cP->Height() << " x " << cP->Width() << '\n';
   // cout << "initial cP " << endl;
   // cout << " ------------------------------------------- " << endl;
   // cP->PrintMatlab();
   // cout << " ------------------------------------------- " << endl;
   // ofstream cp_save("cP.txt");
   // cP->PrintMatlab(cp_save);
   // cp_save.close();
#endif
}

void ParGalerkinDifference::AssembleProlongationMatrix(
    const mfem::Array<int> &id,
    const DenseMatrix &local_mat) const
{
   // element id coresponds to the column indices
   // dofs id coresponds to the row indices
   // the local reconstruction matrix needs to be assembled `vdim` times
   // assume the mesh only contains only 1 type of element
   int nel = id.Size();
   const int main_id_global = id[0];
   const int main_id_local = GetParMesh()->GetLocalElementNum(main_id_global);
   const Element *el = mesh->GetElement(0);
   const FiniteElement *fe =
       fec->FiniteElementForGeometry(el->GetGeometryType());
   const int num_dofs = fe->GetDof();
   MFEM_VERIFY(num_dofs == local_mat.Height(),
               "matrix height doesn't match # of dof");
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
         row_index = el_dofs[v * num_dofs + j];
         HYPRE_IJMatrixSetValues(ij_matrix,
                                 1,
                                 &nel,
                                 &row_index,
                                 col_index.GetData(),
                                 single_row.GetData());
      }
   }
/// serial
#if 0
   int nel = id.Size();
   Array<int> el_dofs;
   Array<int> col_index;
   Array<int> row_index(num_dofs);
   // Array<Array<int>> dofs_mat(vdim);

   // Get the id of the element want to assemble in
   int el_id = id[0];
   GetElementVDofs(el_id, el_dofs);
   // cout << "Element dofs indices are: ";
   // el_dofs.Print(cout, el_dofs.Size());
   // cout << endl;
   // cout << "local mat size is " << el_mat.Height() << ' ' << el_mat.Width()
   // << '\n';
   col_index.SetSize(nel);
   for (int e = 0; e < nel; e++)
   {
      col_index[e] = vdim * id[e];
   }
   for (int v = 0; v < vdim; v++)
   {
      el_dofs.GetSubArray(v * num_dofs, num_dofs, row_index);
      // cout << "local mat will be assembled into: ";
      // row_index.Print(cout, num_dofs);
      // cout << endl;
      cP->SetSubMatrix(row_index, col_index, local_mat, 1);
      row_index.LoseData();
      // elements id also need to be shift accordingly
      col_index.SetSize(nel);
      for (int e = 0; e < nel; e++)
      {
         col_index[e]++;
      }
   }
#endif
}

double ParGalerkinDifference::calcVandScale(int el_id,
                                            int dim,
                                            const DenseMatrix &x_center) const
{
   double dist = -1e+300;
   // get the most further basis
   Vector center(dim);
   GetElementCenter(el_id, center);
   for (int i = 0; i < x_center.Width(); ++i)
   {
      Vector el_center(dim);
      x_center.GetColumn(i, el_center);
      el_center -= center;
      double new_dist = el_center.Norml2();
      if (new_dist > dist)
      {
         dist = new_dist;
      }
   }
   return dist;
}
void ParGalerkinDifference::buildVandermondeMat(int dim,
                                                int num_basis,
                                                const Array<int> &elmt_id,
                                                Array<int> &stencil_elid,
                                                DenseMatrix &x_center,
                                                DenseMatrix &V) const
{
   int el_id = elmt_id[0];
   ofstream cond_file;
   cond_file.open("vand_cond_cut.txt",
                  std::ios_base::app);  // append instead of overwrite
   double cond = 1.0;
   cond = 100.0;  // 100- p2, 1000- p3, 10000- p4
   double vandCond = 1e+30;
   double vand_scale;
   // cout << "elmt_id size: " << elmt_id.Size() << endl;
   /// keep popping off elements until
   // - condition number remains good
   // - or the stencil size becomes minimum required
   stencil_elid.LoseData();
   stencil_elid.Append(el_id);
   // Creat the adjacent array and fill it with the first layer of adj
   // adjcant element list, candidates neighbors, candidates neighbors' adj
   Array<int> adj, cand, cand_adj, cand_next;
   mesh->ElementToElementTable().GetRow(el_id, adj);
   cand.Append(adj);
   int n_ind = 1;
   int nk = stencil_elid.Size();
   while ((vandCond > cond) || (nk < num_basis))
   {
      if (el_id == 365 || el_id == 716 || el_id == 429)
      {
         cout << "adding level: " << n_ind << " neighbors " << endl;
         if (n_ind >= 3)
         {
            cout << "reached level: " << n_ind << " !!!!" << endl;
         }
      }
      for (int i = 0; i < adj.Size(); i++)
      {
         if (-1 == stencil_elid.Find(adj[i]))
         {
            if (embeddedElements.at(adj[i]) == false)
            {
               stencil_elid.Append(adj[i]);
            }
         }
      }
      // cout << "List now is: " << endl;
      // stencil_elid.Print(cout, stencil_elid.Size());
      adj.LoseData();
      for (int i = 0; i < cand.Size(); i++)
      {
         if (embeddedElements.at(cand[i]) == false)
         {
            // cout << "deal with cand " << cand[i];
            mesh->ElementToElementTable().GetRow(cand[i], cand_adj);
            // cout << "'s adj are ";
            // cand_adj.Print(cout, cand_adj.Size());
            for (int j = 0; j < cand_adj.Size(); j++)
            {
               if (-1 == stencil_elid.Find(cand_adj[j]))
               {
                  // cout << cand_adj[j] << " is not found in nels. add to adj
                  // and cand_next.\n";
                  adj.Append(cand_adj[j]);
                  cand_next.Append(cand_adj[j]);
               }
            }
            cand_adj.LoseData();
         }
      }
      cand.LoseData();
      cand = cand_next;
      // cout << "cand copy from next: ";
      // cand.Print(cout, cand.Size());
      cand_next.LoseData();
      // resize the DenseMatrices and clean the data
      x_center.Clear();
      x_center.SetSize(dim, stencil_elid.Size());
      for (int j = 0; j < stencil_elid.Size(); j++)
      {
         // Get and store the element center
         mfem::Vector cent_coord(dim);
         GetElementCenter(stencil_elid[j], cent_coord);
         for (int i = 0; i < dim; i++)
         {
            x_center(i, j) = cent_coord(i);
         }
      }
      vand_scale = calcVandScale(elmt_id[0], dim, x_center);
      int num_elem = x_center.Width();
      // Construct the generalized Vandermonde matrix
      V.SetSize(num_elem, num_basis);
      if (1 == dim)
      {
         for (int i = 0; i < num_elem; ++i)
         {
            double dx = x_center(0, i) - x_center(0, 0);
            for (int p = 0; p <= degree; ++p)
            {
               V(i, p) = pow(dx, p);
            }
         }
      }
      else if (2 == dim)
      {
         for (int i = 0; i < num_elem; ++i)
         {
            double dx = (x_center(0, i) - x_center(0, 0)) / vand_scale;
            double dy = (x_center(1, i) - x_center(1, 0)) / vand_scale;
            int col = 0;
            for (int p = 0; p <= degree; ++p)
            {
               for (int q = 0; q <= p; ++q)
               {
                  V(i, col) = pow(dx, p - q) * pow(dy, q);
                  ++col;
               }
            }
         }
      }
      else if (3 == dim)
      {
         for (int i = 0; i < num_elem; ++i)
         {
            double dx = x_center(0, i) - x_center(0, 0);
            double dy = x_center(1, i) - x_center(1, 0);
            double dz = x_center(2, i) - x_center(2, 0);
            int col = 0;
            for (int p = 0; p <= degree; ++p)
            {
               for (int q = 0; q <= p; ++q)
               {
                  for (int r = 0; r <= p - q; ++r)
                  {
                     V(i, col) = pow(dx, p - q - r) * pow(dy, r) * pow(dz, q);
                     ++col;
                  }
               }
            }
         }
      }
      Vector sv;
      V.SingularValues(sv);
      vandCond = sv(0) / sv(sv.Size() - 1);
      nk = stencil_elid.Size();
      ++n_ind;
   }
   if (el_id == 365 || el_id == 716 || el_id == 429)
   {
      cout << " num_el " << stencil_elid.Size() << endl;
      cout << " ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ " << endl;
      cout << " vandCond final " << vandCond << endl;
      cout << " ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ " << endl;
      cond_file << el_id << " : " << vandCond << "\n";
   }
}
#if 0
void ParGalerkinDifference::buildVandermondeMat(int dim,
                                                int num_basis,
                                                const Array<int> &elmt_id,
                                                Array<int> &stencil_elid,
                                                DenseMatrix &x_center,
                                                DenseMatrix &V) const
{
   int el_id = elmt_id[0];
   ofstream cond_file;
   cond_file.open("vand_cond_cut.txt",
                  std::ios_base::app);  // append instead of overwrite
   // cout << "#elements in initial stencil " << elmt_id.Size() << endl;
   // elmt_id.Print(cout, elmt_id.Size());
   double cond = 1.0;
   cond = 2.0;
   double vandCond = 1e+30;
   // int num_el = num_basis;
   int num_el = elmt_id.Size();
   double vand_scale;
   // cout << "elmt_id size: " << elmt_id.Size() << endl;
   /// keep popping off elements until
   // - condition number remains good
   // - or the stencil size becomes minimum required
   stencil_elid.LoseData();
   while (vandCond > cond || num_el == num_basis)
   {
      if (num_el > num_basis)
      {
         if (num_el == elmt_id.Size())
         {
            for (int k = 0; k < num_el; ++k)
            {
               stencil_elid.Append(elmt_id[k]);
            }
         }
         else
         {
            stencil_elid.DeleteLast();
         }
         // resize the DenseMatrices and clean the data
         x_center.Clear();
         x_center.SetSize(dim, num_el);
         for (int j = 0; j < num_el; j++)
         {
            // Get and store the element center
            mfem::Vector cent_coord(dim);
            GetElementCenter(elmt_id[j], cent_coord);
            for (int i = 0; i < dim; i++)
            {
               x_center(i, j) = cent_coord(i);
            }
         }
         vand_scale = calcVandScale(elmt_id[0], dim, x_center);
         int num_elem = x_center.Width();
         // Construct the generalized Vandermonde matrix
         V.SetSize(num_elem, num_basis);
         if (1 == dim)
         {
            for (int i = 0; i < num_elem; ++i)
            {
               double dx = x_center(0, i) - x_center(0, 0);
               for (int p = 0; p <= degree; ++p)
               {
                  V(i, p) = pow(dx, p);
               }
            }
         }
         else if (2 == dim)
         {
            for (int i = 0; i < num_elem; ++i)
            {
               double dx = (x_center(0, i) - x_center(0, 0)) / vand_scale;
               double dy = (x_center(1, i) - x_center(1, 0)) / vand_scale;
               int col = 0;
               for (int p = 0; p <= degree; ++p)
               {
                  for (int q = 0; q <= p; ++q)
                  {
                     V(i, col) = pow(dx, p - q) * pow(dy, q);
                     ++col;
                  }
               }
            }
         }

         else if (3 == dim)
         {
            for (int i = 0; i < num_elem; ++i)
            {
               double dx = x_center(0, i) - x_center(0, 0);
               double dy = x_center(1, i) - x_center(1, 0);
               double dz = x_center(2, i) - x_center(2, 0);
               int col = 0;
               for (int p = 0; p <= degree; ++p)
               {
                  for (int q = 0; q <= p; ++q)
                  {
                     for (int r = 0; r <= p - q; ++r)
                     {
                        V(i, col) =
                            pow(dx, p - q - r) * pow(dy, r) * pow(dz, q);
                        ++col;
                     }
                  }
               }
            }
         }

         Vector sv;
         V.SingularValues(sv);
         vandCond = sv(0) / sv(sv.Size() - 1);
         --num_el;
      }
#if 0
      if (num_el <= elmt_id.Size())
      {
         if (num_el == num_basis)
         {
            for (int k = 0; k < num_el; ++k)
            {
               stencil_elid.Append(elmt_id[k]);
            }
         }
         else
         {
            stencil_elid.Append(elmt_id[num_el - 1]);
         }

         // resize the DenseMatrices and clean the data
         x_center.Clear();
         x_center.SetSize(dim, num_el);
         for (int j = 0; j < num_el; j++)
         {
            // Get and store the element center
            mfem::Vector cent_coord(dim);
            GetElementCenter(elmt_id[j], cent_coord);
            for (int i = 0; i < dim; i++)
            {
               x_center(i, j) = cent_coord(i);
            }
         }
         vand_scale = calcVandScale(elmt_id[0], dim, x_center);
         // vand_scale = 1.0;
         int num_elem = x_center.Width();
         // Construct the generalized Vandermonde matrix
         V.SetSize(num_elem, num_basis);
         if (1 == dim)
         {
            for (int i = 0; i < num_elem; ++i)
            {
               double dx = x_center(0, i) - x_center(0, 0);
               for (int p = 0; p <= degree; ++p)
               {
                  V(i, p) = pow(dx, p);
               }
            }
         }
         else if (2 == dim)
         {
            for (int i = 0; i < num_elem; ++i)
            {
               double dx = (x_center(0, i) - x_center(0, 0)) / vand_scale;
               double dy = (x_center(1, i) - x_center(1, 0)) / vand_scale;
               int col = 0;
               for (int p = 0; p <= degree; ++p)
               {
                  for (int q = 0; q <= p; ++q)
                  {
                     V(i, col) = pow(dx, p - q) * pow(dy, q);
                     ++col;
                  }
               }
            }
         }

         else if (3 == dim)
         {
            for (int i = 0; i < num_elem; ++i)
            {
               double dx = x_center(0, i) - x_center(0, 0);
               double dy = x_center(1, i) - x_center(1, 0);
               double dz = x_center(2, i) - x_center(2, 0);
               int col = 0;
               for (int p = 0; p <= degree; ++p)
               {
                  for (int q = 0; q <= p; ++q)
                  {
                     for (int r = 0; r <= p - q; ++r)
                     {
                        V(i, col) =
                            pow(dx, p - q - r) * pow(dy, r) * pow(dz, q);
                        ++col;
                     }
                  }
               }
            }
         }

         Vector sv;
         V.SingularValues(sv);

         // Vector cent10(dim);
         // GetElementCenter(elem_id, cent10);
         // cent10.Print();
         // V.PrintMatlab();
         // cout << "matrix of element centers is: " << endl;
         // x_center.PrintMatlab();
         vandCond = sv(0) / sv(sv.Size() - 1);
         ++num_el;
      }
#endif
      else
      {
         break;
      }
   }

   cout << "num_el " << num_el << endl;
   cout << " ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ " << endl;
   cout << " vandCond final " << vandCond << endl;
   cout << " ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ " << endl;
   cond_file << el_id << " : " << vandCond << "\n";

}
#endif

void ParGalerkinDifference::buildLSInterpolation(int elem_id,
                                                 int dim,
                                                 int degree,
                                                 const DenseMatrix &V,
                                                 const DenseMatrix &x_center,
                                                 const DenseMatrix &x_quad,
                                                 DenseMatrix &interp) const
{
   // get the number of quadrature points and elements.
   int num_quad = x_quad.Width();
   int num_elem = x_center.Width();
   double vand_scale = calcVandScale(elem_id, dim, x_center);
   // cout << "vand_scale for elem_id " << elem_id << " is " << vand_scale <<
   // endl;
   // vand_scale = 1.0;
   // number of total polynomial basis functions
   int num_basis = -1;
   if (1 == dim)
   {
      num_basis = degree + 1;
   }
   else if (2 == dim)
   {
      num_basis = (degree + 1) * (degree + 2) / 2;
   }
   else if (3 == dim)
   {
      num_basis = (degree + 1) * (degree + 2) * (degree + 3) / 6;
   }
   else
   {
      cout << "buildLSInterpolation: dim must be 3 or less.\n" << endl;
   }

// ---------------------------------------------------------------------------
/// use this for triangle elements
// Set the RHS for the LS problem (it's the identity matrix)
// This will store the solution, that is, the basis coefficients, hence
// the name `coeff`
#if 0
   mfem::DenseMatrix coeff(num_elem, num_elem);
   coeff = 0.0;
   for (int i = 0; i < num_elem; ++i)
   {
      coeff(i, i) = 1.0;
   }
   // Set-up and solve the least-squares problem using LAPACK's dgels
   char TRANS = 'N';
   int info;
   int lwork = 2 * num_elem * num_basis;
   double work[lwork];
   dgels_(&TRANS,
          &num_elem,
          &num_basis,
          &num_elem,
          V.GetData(),
          &num_elem,
          coeff.GetData(),
          &num_elem,
          work,
          &lwork,
          &info);
   MFEM_ASSERT(info == 0, "Fail to solve the underdetermined system.\n");
#endif
// -------------------------------------------------------------------------------

/// use this for quad elements
#if 1
   // Set the RHS for the LS problem (it's the identity matrix)
   // This will store the solution, that is, the basis coefficients, hence
   // the name `coeff`
   int LDB = max(num_elem, num_basis);
   // cout << "LDB: " << LDB << endl;
   mfem::DenseMatrix coeff(LDB, LDB);
   coeff = 0.0;
   for (int i = 0; i < LDB; ++i)
   {
      coeff(i, i) = 1.0;
   }

   // Set-up and solve the least-squares problem using LAPACK's dgels
   char TRANS = 'N';
   int info;
   int lwork = (num_elem * num_basis) + (3 * num_basis) + 1;
   double work[lwork];
   int rank;
   Array<int> jpvt;
   jpvt.SetSize(num_basis);
   jpvt = 0;
   double rcond = 1e-16;

   dgelsy_(&num_elem,
           &num_basis,
           &num_elem,
           V.GetData(),
           &num_elem,
           coeff.GetData(),
           &LDB,
           jpvt.GetData(),
           &rcond,
           &rank,
           work,
           &lwork,
           &info);

   MFEM_ASSERT(info == 0, "Fail to solve the underdetermined system.\n");
#endif
   // Perform matrix-matrix multiplication between basis functions evalauted at
   // quadrature nodes and basis function coefficients.
   interp.SetSize(num_quad, num_elem);
   interp = 0.0;
   if (1 == dim)
   {
      // loop over quadrature points
      for (int j = 0; j < num_quad; ++j)
      {
         double dx = x_quad(0, j) - x_center(0, 0);
         // loop over the element centers
         for (int i = 0; i < num_elem; ++i)
         {
            for (int p = 0; p <= degree; ++p)
            {
               interp(j, i) += pow(dx, p) * coeff(p, i);
            }
         }
      }
   }
   else if (2 == dim)
   {
      // loop over quadrature points
      for (int j = 0; j < num_quad; ++j)
      {
         double dx = (x_quad(0, j) - x_center(0, 0)) / vand_scale;
         double dy = (x_quad(1, j) - x_center(1, 0)) / vand_scale;
         // loop over the element centers
         for (int i = 0; i < num_elem; ++i)
         {
            int col = 0;
            for (int p = 0; p <= degree; ++p)
            {
               for (int q = 0; q <= p; ++q)
               {
                  interp(j, i) += pow(dx, p - q) * pow(dy, q) * coeff(col, i);
                  ++col;
               }
            }
         }
      }
      // loop over quadrature points
      for (int j = 0; j < num_quad; ++j)
      {
         for (int p = 0; p <= degree; ++p)
         {
            for (int q = 0; q <= p; ++q)
            {
               // loop over the element centers
               double poly_at_quad = 0.0;
               for (int i = 0; i < num_elem; ++i)
               {
                  double dx = (x_quad(0, j) - x_center(0, i)) / vand_scale;
                  double dy = (x_quad(1, j) - x_center(1, i)) / vand_scale;
                  poly_at_quad += interp(j, i) * pow(dx, p - q) * pow(dy, q);
               }
               double exact = ((p == 0) && (q == 0)) ? 1.0 : 0.0;
               // mfem::out << "polynomial interpolation error (" << p - q <<
               // ","
               //           << q << ") = " << fabs(exact - poly_at_quad) <<
               //           endl;
               // if ((p == 0) && (q == 0))
               // {
               MFEM_ASSERT(fabs(exact - poly_at_quad) <= 1e-12,
                           " p = " << p << " , q = " << q << ", "
                                   << fabs(exact - poly_at_quad) << " : "
                                   << "Interpolation operator does not "
                                      "interpolate exactly!\n");
               // }
            }
         }
      }
   }
   else if (dim == 3)
   {
      // loop over quadrature points
      for (int j = 0; j < num_quad; ++j)
      {
         double dx = x_quad(0, j) - x_center(0, 0);
         double dy = x_quad(1, j) - x_center(1, 0);
         double dz = x_quad(2, j) - x_center(2, 0);
         // loop over the element centers
         for (int i = 0; i < num_elem; ++i)
         {
            int col = 0;
            for (int p = 0; p <= degree; ++p)
            {
               for (int q = 0; q <= p; ++q)
               {
                  for (int r = 0; r <= p - q; ++r)
                  {
                     interp(j, i) += pow(dx, p - q - r) * pow(dy, r) *
                                     pow(dz, q) * coeff(col, i);
                     ++col;
                  }
               }
            }
         }
      }
   }
}