#include <fstream>
#include <iostream>
#include "default_options.hpp"
#include "galer_diff.hpp"
using namespace std;
using namespace mach;
using namespace apf;

namespace mfem
{

ParGDSpace::ParGDSpace(Mesh *m, ParMesh *pm, const FiniteElementSpace *global_fes,
                       const int *partitioning, const FiniteElementCollection *fec,
                       int vdim, int ordering, int de, int p)
   : pr(p), full_mesh(m), full_fespace(global_fes), SpaceType(pm,global_fes,partitioning)
{
   degree = de;
   total_nel = full_mesh->GetNE();

   if (GetMyRank() == pr)
   {
      BuildProlongationOperator();
   }

   local_tdof = vdim * GetParMesh()->GetNE();
   MPI_Allreduce(&local_tdof, &total_tdof, 1, HYPRE_MPI_INT, MPI_SUM, GetComm());

   HYPRE_BigInt row_starts[2], col_starts[GetNRanks()+1];


   col_starts[0] = 0;
   col_starts[1] = 8;
   col_starts[2] = 16;
   col_starts[3] = 24;
   col_starts[4] = 32;
   row_starts[0] = 0;
   row_starts[1] = 224;


   if (GetMyRank() == pr)
   {
      cout << "row start and end are " << row_starts[0] << ", " << row_starts[1] << endl;
      cout << "col start and end are " << col_starts[0] << ", " << col_starts[1] << endl;
   }

   MPI_Barrier(GetComm());

   // hypre_CSRMatrix *csr_a;
   // csr_a = hypre_CSRMatrixCreate(GetCP()->Height(), GetCP()->Width(),GetCP()->NumNonZeroElems());
   // hypre_CSRMatrixSetDataOwner(csr_a,1);
   // hypre_CSRMatrixI(csr_a) = GetCP()->GetI();
   // hypre_CSRMatrixJ(csr_a) = GetCP()->GetJ();
   // hypre_CSRMatrixData(csr_a) = GetCP()->GetData();
   // hypre_CSRMatrixSetRownnz(csr_a);

   // hypre_ParCSRMatrix *par_csr;
   // // par_csr = hypre_CSRMatrixToParCSRMatrix(GetComm(),(HYPRE_CSRMatrix)csr_a,row_starts,col_starts);
   // par_csr = hypre_CSRMatrixToParCSRMatrix(GetComm(), csr_a, row_starts, col_starts);
   // // hypre_ParCSRMatrix * aa;
   // // aa = DistributeGloballyReplicatedMatrix(GetComm(),GetCP()->GetI(),GetCP()->GetJ(),
   // //                                         GetCP()->GetData(), row_starts, col_starts);
   P = new HypreParMatrix(GetComm(), row_starts, col_starts, GetCP());
   cout << "HypreProlongation matrix size are " << P->Height() << " x " << P->Width() << endl;
}

void ParGDSpace::GetNeighbourSet(int id, int req_n,
                                 mfem::Array<int> &nels)
{
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

   for(int j = 0; j < num_el; j++)
   {
      // Get and store the element center
      full_mesh->GetElementCenter(elmt_id[j], cent_coord);
   

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
   // assume the mesh only contains only 1 type of element
   const Element* el = full_mesh->GetElement(0);
   const FiniteElement *fe = fec->FiniteElementForGeometry(el->GetGeometryType());
   const int num_dofs = fe->GetDof();
   const int dim = full_mesh->Dimension();

   cP = new mfem::SparseMatrix(full_fespace->GetVSize(), vdim * total_nel);
   // determine the minimum # of element in each patch
   int nelmt;
   switch(full_mesh->Dimension())
   {
      case 1: nelmt = degree + 1; break;
      case 2: nelmt = (degree+1) * (degree+2) / 2; break;
      case 3: throw MachException("Not implemeneted yet.\n"); break;
      default: throw MachException("dim must be 1, 2 or 3.\n");
   }

   cout << "cp size is " << cP->Height() << " x " << cP->Width() << '\n';
   // vector that contains element id (resize to zero )
   mfem::Array<int> elmt_id;
   mfem::DenseMatrix cent_mat, quad_mat, local_mat;
   for (int i = 0; i < total_nel; i++)
   {
      // 1. Get element id in patch
      GetNeighbourSet(i, nelmt, elmt_id);
      // cout << "id(s) in patch " << i << ": ";
      // elmt_id.Print(cout, elmt_id.Size());

      // 2. build the quadrature and barycenter coordinate matrices
      BuildNeighbourMat(elmt_id, cent_mat, quad_mat);
      // cout << "The element center matrix:\n";
      // cent_mat.Print(cout, cent_mat.Width());
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
   cP->Finalize();
   cP_is_set = true;
   ofstream cp_save("cP.txt");
   cP->PrintMatlab(cp_save);
   cp_save.close();
}

void ParGDSpace::AssembleProlongationMatrix(const mfem::Array<int> &id,
                                            const DenseMatrix &local_mat) const
{
   // element id coresponds to the column indices
   // dofs id coresponds to the row indices
   // the local reconstruction matrix needs to be assembled `vdim` times
   // assume the mesh only contains only 1 type of element
   const Element* el = full_mesh->GetElement(0);
   const FiniteElement *fe = fec->FiniteElementForGeometry(el->GetGeometryType());
   const int num_dofs = fe->GetDof();

   int nel = id.Size();
   Array<int> el_dofs;
   Array<int> col_index(nel);
   Array<int> row_index(num_dofs);

   int el_id = id[0];
   full_fespace->GetElementVDofs(el_id, el_dofs);
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

hypre_ParCSRMatrix* ParGDSpace::DistributeGloballyReplicatedMatrix(
   MPI_Comm comm, HYPRE_Int* serial_I, HYPRE_Int* serial_J,
   HYPRE_Complex* serial_Data, HYPRE_Int* my_row_starts, HYPRE_Int* my_col_starts)
{

   const HYPRE_Int global_num_rows = my_row_starts[2],
      global_num_cols = my_col_starts[2];

   const HYPRE_Int my_first_row = my_row_starts[0],
      my_last_row = my_row_starts[1];

   const HYPRE_Int my_first_col = my_col_starts[0],
      my_last_col = my_col_starts[1];

   const HYPRE_Int my_num_rows = my_last_row - my_first_row;


   // Create the offd_col_map first
   HYPRE_Int * global_idx_marker =
      (HYPRE_Int *) malloc(global_num_cols*sizeof(HYPRE_Int));

   for (int i=0;i<global_num_cols;i++)
      global_idx_marker[i] =  0;

   HYPRE_Int num_shared_cols = 0, diag_nnz = 0, offd_nnz = 0;
   for (HYPRE_Int index = serial_I[my_first_row];
        index < serial_I[my_last_row];
        ++index)
   {
      const HYPRE_Int& the_global_column_index = serial_J[index];

      if ( (the_global_column_index < my_first_col) ||
           (the_global_column_index >= my_last_col) )
      {
         // The index is outside my owned range; mark and up count
         if (!global_idx_marker[the_global_column_index])
            global_idx_marker[the_global_column_index] = ++num_shared_cols;

         // Up nnz count
         offd_nnz++;
      }
      else
         diag_nnz++;
   }

   // Create the new matrix
   hypre_ParCSRMatrix * parallel_matrix = hypre_ParCSRMatrixCreate(
      comm,global_num_rows,global_num_cols,
      my_row_starts,my_col_starts,
      num_shared_cols,diag_nnz,offd_nnz);

   // In my case, I keep the partitioning arrays
   hypre_ParCSRMatrixOwnsRowStarts(parallel_matrix) = 0;
   hypre_ParCSRMatrixOwnsColStarts(parallel_matrix) = 0;

   // Initialize the matrix -- allocates Diag and Offd
   hypre_ParCSRMatrixInitialize(parallel_matrix);

   // Fill col_map_offd
   HYPRE_Int * col_map_offd = hypre_ParCSRMatrixColMapOffd(parallel_matrix);
   num_shared_cols = 0;
   for (size_t global_idx = 0;
        global_idx < global_num_cols;
        ++global_idx)
   {
      if (global_idx_marker[global_idx] > 0)
      {
         col_map_offd[num_shared_cols] = global_idx;
         global_idx_marker[global_idx] = ++num_shared_cols;
         // Note: this will mark each shared global idx with a
         // positive number, so
         //
         // local_idx = global_idx_marker[global_idx] - 1.
      }
   }


   // Copy the matrix data
   hypre_CSRMatrix * diag = hypre_ParCSRMatrixDiag(parallel_matrix),
      * offd = hypre_ParCSRMatrixOffd(parallel_matrix);

   HYPRE_Int * diag_i = hypre_CSRMatrixI(diag),
      * diag_j = hypre_CSRMatrixJ(diag);
   HYPRE_Complex * diag_data = hypre_CSRMatrixData(diag);

   HYPRE_Int * offd_i = hypre_CSRMatrixI(offd),
      * offd_j = hypre_CSRMatrixJ(offd);
   HYPRE_Complex * offd_data = hypre_CSRMatrixData(offd);

   diag_nnz = 0; offd_nnz = 0;
   for (HYPRE_Int global_row = my_first_row, local_row = 0;
        global_row < my_last_row;++global_row,++local_row)
   {
      diag_i[local_row] = diag_nnz;
      offd_i[local_row] = offd_nnz;

      for (HYPRE_Int j_idx = serial_I[global_row];
           j_idx < serial_I[global_row+1]; ++j_idx)
      {
         const HYPRE_Int& global_idx = serial_J[j_idx];

         if ((global_idx < my_first_col) || (global_idx >= my_last_col))
         {
            // Column is shared, add to offd
            offd_j[offd_nnz] = global_idx_marker[global_idx] - 1;
            offd_data[offd_nnz++] = serial_Data[j_idx];
         }
         else
         {
            // Column is owned, add to diag
            diag_j[diag_nnz] = global_idx - my_first_col;
            diag_data[diag_nnz++] = serial_Data[j_idx];
         }
      }
   }
   diag_i[my_num_rows] = diag_nnz;
   offd_i[my_num_rows] = offd_nnz;

   // Hypre wants the diagonal entry to come first.
   HYPRE_Complex tmp_data;
   for (HYPRE_Int local_row=0; local_row < my_num_rows; ++local_row)
   {
      HYPRE_Int first_col = diag_i[local_row];
      for (HYPRE_Int j_idx = first_col; j_idx < diag_i[local_row+1]; ++j_idx)
      {
         if (diag_j[j_idx] == local_row)
         {
            // Swap the column indices
            diag_j[j_idx] = diag_j[first_col];
            diag_j[first_col] = local_row;

            // Swap the data
            tmp_data = diag_data[j_idx];
            diag_data[j_idx] = diag_data[first_col];
            diag_data[first_col] = tmp_data;
            break;
         }
      }
   }

   // Now the hypre_ParCSRMatrix should be built. Create the commpkg:
   hypre_MatvecCommPkgCreate(parallel_matrix);

   free(global_idx_marker);

   return parallel_matrix;
}

// ParGDSpace::ParGDSpace(Mesh2 *apf_mesh, ParMesh *pm, const FiniteElementCollection *f,
//                     	  int vdim, int ordering, int de, int p)
//    : pr(p), SpaceType(pm, f, vdim, ordering)
// {
//    pumi_mesh = apf_mesh;
//    dim = pm->Dimension();
//    degree = de;
//    local_tdof = vdim * GetParMesh()->GetNE();
//    cout << "Before calculate element offsts.\n";
//    GetParMesh()->ComputeGlobalElementOffset(); // something wrong here, for 1 processor probably works.
//    cout << "After calculate element offsts.\n";
//    MPI_Allreduce(&local_tdof, &total_tdof, 1, HYPRE_MPI_INT, MPI_SUM, GetComm());
//    // dimension set up of the prolongation operator
//    dof_start = (GetTrueDofOffsets())[0];
//    dof_end = (GetTrueDofOffsets())[1] - 1;
//    el_start = GetParMesh()->GetElementOffset();
//    el_end = el_start + local_tdof - 1;

//    BuildProlongationOperator();

//    const char* p_save = "prolong";
//    dynamic_cast<HypreParMatrix*>(P)->Print(p_save);
//    if (pr == GetMyRank())
//    {
//       cout << "\n-------"
//            << " ParGD Info "
//            << "-------\n";
//       cout << setw(15) << "Processor id:" << setw(8) << pr << '\n';
//       cout << setw(15) << "vdim is:" << setw(8) << vdim << '\n';
//       cout << setw(15) << "row start:" << setw(8) << dof_start << '\n';
//       cout << setw(15) << "row end:" << setw(8) << dof_end << '\n';
//       cout << setw(15) << "column start:" << setw(8) << el_start << '\n';
//       cout << setw(15) << "column end:" << setw(8) << el_end << '\n';
//       cout << setw(15) << "global  vdof:" << setw(8) << GlobalVSize() << '\n';
//       cout << setw(15) << "global tdof:" << setw(8) << GlobalTrueVSize() << '\n';
//       cout << setw(15) << "P size:" << setw(8) << dynamic_cast<HypreParMatrix*>(P)->Height() << " x "
//         << dynamic_cast<HypreParMatrix*>(P)->Width() << '\n';
//       cout << "--------------------------\n";
//    }
// }

// version 1
// void ParGDSpace::BuildProlongationOperator()
// {
//    // initialize the hypre par matrix 
//    HYPRE_IJMatrixCreate(GetComm(), dof_start, dof_end,
//                         el_start, el_end, &ij_matrix);
//    HYPRE_IJMatrixSetObjectType(ij_matrix, HYPRE_PARCSR);
//    HYPRE_IJMatrixInitialize(ij_matrix);
//    // The parallel prolongation matrix is built as a HypreParMatrix
//    // follow the hypre API
//    // assume the mesh only contains only 1 type of element
//    const Element* el = GetParMesh()->GetElement(0);
//    const FiniteElement *fe = fec->FiniteElementForGeometry(el->GetGeometryType());
//    const int num_dofs = fe->GetDof();
//    // allocate the space for the prolongation matrix
//    // this step should be done in the constructor (probably)
//    // should it be GetTrueVSize() ? or GetVSize()?
//    // need a new method that directly construct a CSR format sparsematrix ？
//    // cP = new mfem::SparseMatrix(GetVSize(), vdim * nEle);
//    // determine the minimum # of element in each patch
//    int nelmt;
//    switch(dim)
//    {
//       case 1: nelmt = degree + 1; break;
//       case 2: nelmt = (degree+1) * (degree+2) / 2; break;
//       case 3: throw MachException("Not implemeneted yet.\n"); break;
//       default: throw MachException("dim must be 1, 2 or 3.\n");
//    }
//    // cout << "Number of required element: " << nelmt << '\n';
//    // loop over all the element:
//    // 1. build the patch for each element,
//    // 2. construct the local reconstruction operator
//    // 3. assemble local reconstruction operator
   
//    // vector that contains element id (resize to zero )
//    Array<int> elmt_id;
//    mfem::DenseMatrix cent_mat, quad_mat, local_mat;
//    //cout << "The size of the prolongation matrix is " << cP->Height() << " x " << cP->Width() << '\n';
//    //int degree_actual;
//    for (int i = 0; i < GetParMesh()->GetNE(); i++)
//    {
//       // 1. construct the patch the patch
//       GetNeighbourSet(i, nelmt, elmt_id);
//       cout << "Element id in patch " << i << ": ";
//       elmt_id.Print(cout, elmt_id.Size());
      
//       // 2. build the quadrature and barycenter coordinate matrices
//       GetNeighbourMat(elmt_id, cent_mat, quad_mat);
//       // cout << "The element center matrix:\n";
//       // cent_mat.Print(cout, cent_mat.Width());
//       // cout << endl;
//       // cout << "Quadrature points id matrix:\n";
//       // quad_mat.Print(cout, quad_mat.Width());
//       // cout << endl;

//       // 3. buil the loacl reconstruction matrix
//       buildLSInterpolation(dim, degree, cent_mat, quad_mat, local_mat);
//       // buildInterpolation(dim, degree, cent_mat, quad_mat, local_mat);
//       // cout << "Local reconstruction matrix R:\n";
//       // local_mat.Print(cout, local_mat.Width());
//       // cout << "row-wise? " << *(local_mat.GetData()) << ' ' << *(local_mat.GetData()+1) << '\n'; 

//       // 4. assemble them back to prolongation matrix
//       AssembleProlongationMatrix(elmt_id, local_mat);
//    }
//    // finialize the hypre parmatrix
//    HYPRE_IJMatrixAssemble(ij_matrix);
//    HYPRE_IJMatrixGetObject(ij_matrix, (void**)&prolong);

//    P = new HypreParMatrix((hypre_ParCSRMatrix*)(prolong), true);

//    Vector diag(local_tdof);
//    diag = 1.0;
//    R = new SparseMatrix(diag);
// }

// version 2 that use sparse matrix to construct the parallel gd matrix
// void ParGDSpace::BuildProlongationOperator()
// {
//    // The parallel prolongation matrix is built as a HypreParMatrix
//    // follow the hypre API
//    // assume the mesh only contains only 1 type of element
//    const Element* el = GetParMesh()->GetElement(0);
//    const FiniteElement *fe = fec->FiniteElementForGeometry(el->GetGeometryType());
//    const int num_dofs = fe->GetDof();
//    // allocate the space for the prolongation matrix
//    // this step should be done in the constructor (probably)
//    // should it be GetTrueVSize() ? or GetVSize()?
//    // need a new method that directly construct a CSR format sparsematrix ？
//    cP = new mfem::SparseMatrix(GetVSize(), vdim * nEle);
//    // determine the minimum # of element in each patch
//    int nelmt;
//    switch(dim)
//    {
//       case 1: nelmt = degree + 1; break;
//       case 2: nelmt = (degree+1) * (degree+2) / 2; break;
//       case 3: throw MachException("Not implemeneted yet.\n"); break;
//       default: throw MachException("dim must be 1, 2 or 3.\n");
//    }
//    // cout << "Number of required element: " << nelmt << '\n';
//    // loop over all the element:
//    // 1. build the patch for each element,
//    // 2. construct the local reconstruction operator
//    // 3. assemble local reconstruction operator
   
//    // vector that contains element id (resize to zero )
//    Array<int> elmt_id;
//    mfem::DenseMatrix cent_mat, quad_mat, local_mat;
//    //cout << "The size of the prolongation matrix is " << cP->Height() << " x " << cP->Width() << '\n';
//    //int degree_actual;
//    for (int i = 0; i < GetParMesh()->GetNE(); i++)
//    {
//       // 1. construct the patch the patch
//       GetNeighbourSet(i, nelmt, elmt_id);
//       // cout << "Element id in patch " << i << ": ";
//       // elmt_id.Print(cout, elmt_id.Size());
      
//       // 2. build the quadrature and barycenter coordinate matrices
//       GetNeighbourMat(elmt_id, cent_mat, quad_mat);
//       // cout << "The element center matrix:\n";
//       // cent_mat.Print(cout, cent_mat.Width());
//       // cout << endl;
//       // cout << "Quadrature points id matrix:\n";
//       // quad_mat.Print(cout, quad_mat.Width());
//       // cout << endl;

//       // 3. buil the loacl reconstruction matrix
//       buildLSInterpolation(dim, degree, cent_mat, quad_mat, local_mat);
//       // cout << "Local reconstruction matrix R:\n";
//       // local_mat.Print(cout, local_mat.Width());

//       // 4. assemble them back to prolongation matrix
//       AssembleProlongationMatrix(elmt_id, local_mat);
//    }
//    // finialize the hypre parmatrix
//    cP->Finalize();
//    cP_is_set = true;
//    cout << "Check cP size: " << cP->Height() << " x " << cP->Width() << '\n';
//    P = new HypreParMatrix(GetComm(), );
// }

// void ParGDSpace::GetNeighbourSet(int id, int req_n, Array<int> &nels)
// {
//    /// this stores the elements for which we need neighbours
//    vector<pMeshEnt> el;
//    pMeshEnt e;
//    /// get pumi mesh entity (element) for the given id
//    e = apf::getMdsEntity(pumi_mesh, dim, id);
//    /// first, need to find neighbour of the given element
//    el.push_back(e);
//    /// first entry in neighbour vector should be the element itself
//    nels.SetSize(0); // clean the queue vector 
//    nels.Append(id);
//    /// iterate for finding element neighbours.
//    /// it stops when the # of elements in patch are equal/greater
//    /// than the minimum required # of elements in patch.
//    while (nels.Size() < req_n)
//    {
//       /// this stores the neighbour elements for which we need neighbours
//       vector<pMeshEnt> elm;
//       ///get neighbours (with shared edges)
//       for (int j = 0; j < el.size(); ++j)
//       {
//          /// vector for storing neighbours of el[j]
//          Adjacent nels_e1;
//          /// get neighbours
//          getBridgeAdjacent(pumi_mesh, el[j], pumi_mesh_getDim(pumi_mesh) - 1,
//                            pumi_mesh_getDim(pumi_mesh), nels_e1);
//          /// retrieve the id of neighbour elements
//          /// push in nels
//          for (int i = 0; i < nels_e1.size(); ++i)
//          {
//             int nid;
//             nid = getMdsIndex(pumi_mesh, nels_e1[i]);
//             /// check for element, push it if not there already
//             if (!(std::find(nels.begin(), nels.end(), nid) != nels.end()))
//             {
//                nels.Append(nid);
//             }
//             /// push neighbour elements for next iteration
//             /// and use them if required
//             elm.push_back(nels_e1[i]);
//          }
//       }
//       /// resizing el to zero prevents finding neighbours of the same elements
//       el.resize(0);
//       /// insert the neighbour elements in 'el' and iterate to find their neighbours if needed
//       el.insert(end(el), begin(elm), end(elm));
//    }
// }

// void ParGDSpace::GetNeighbourMat(Array<int> &els_id, DenseMatrix &mat_cent,
//                                 DenseMatrix &mat_quad) const
// {
//    // assume the mesh only contains only 1 type of element
//    const Element* el = GetParMesh()->GetElement(0);
//    const FiniteElement *fe = fec->FiniteElementForGeometry(el->GetGeometryType());
//    const int num_dofs = fe->GetDof();

//    // resize the DenseMatrices and clean the data
//    int num_el = els_id.Size();
//    mat_cent.Clear(); 
//    mat_cent.SetSize(dim, num_el);
//    mat_quad.Clear();
//    mat_quad.SetSize(dim, num_dofs);
//    // vectors that hold coordinates of quadrature points
//    // used for duplication tests
//    vector<double> quad_data;
//    Vector quad_coord(dim); // used to store quadrature coordinate temperally
//    Vector cent_coord(dim);
//    for(int j = 0; j < num_el; j++)
//    {
//       // Get and store the element center
//       GetElementCenter(els_id[j], cent_coord);
//       for(int i = 0; i < dim; i++)
//       {
//          mat_cent(i,j) = cent_coord(i);
//       }
//    }

//    // deal with quad points
//    ElementTransformation *eltransf = GetParMesh()->GetElementTransformation(els_id[0]);
//    for (int j = 0; j < num_dofs; j++)
//    {
//       eltransf->Transform(fe->GetNodes().IntPoint(j), quad_coord);
//       for (int i = 0; i < dim; i++)
//       {
//          mat_quad(i, j) = quad_coord(i);
//       }
//    }
// }

// version 1
// void ParGDSpace::AssembleProlongationMatrix(const Array<int> &els_id,
//                                             DenseMatrix &local_mat)
// {
//    const Element *el = GetParMesh()->GetElement(0);
//    const FiniteElement *fe = fec->FiniteElementForGeometry(el->GetGeometryType());
//    const int num_dofs = fe->GetDof();
//    // need to transpose the local_mat given the data format
//    local_mat.Transpose();
   
//    int nel = els_id.Size();
//    Array<int> el_dofs;
//    Array<int> col_index(nel*num_dofs);
//    Array<int> row_index(num_dofs);
 
//    int el_id = els_id[0];
//    GetElementVDofs(el_id, el_dofs);
//    Array<int> ncols(num_dofs);
//    ncols = nel;
//    // cout << "ncols is: "; ncols.Print(cout, ncols.Size());
//    // cout << "mat size is " << local_mat.Height() << " x " << local_mat.Width() << '\n';
//    // cout << "col_index size is " << col_index.Size() << '\n';
//    for (int d = 0; d < num_dofs; d++)
//    {
//       for (int e = 0; e < nel; e++)
//       {
//          col_index[d * nel + e] = vdim * els_id[e];
//       }
//    }
//    for (int v = 0; v < vdim; v++)
//    {
//       el_dofs.GetSubArray(v * num_dofs, num_dofs, row_index);
//       cout << '\t' <<  v << " assemble location:\n";
//       cout << "\t\tRow:"; row_index.Print(cout, row_index.Size());
//       cout << "\t\tCol:"; col_index.Print(cout, col_index.Size());
//       HYPRE_IJMatrixSetValues(ij_matrix, num_dofs, ncols.GetData(), row_index.GetData(),
//                              col_index.GetData(), local_mat.GetData());
//       row_index.LoseData();
//       for (int e = 0; e < nel * num_dofs; e++)
//       {
//          col_index[e]++;
//       }
//    }
// }

// HypreParVector *ParGDSpace::NewTrueDofVector()
// {
//    std::cout << "ParGDSpace::NewTrueDofVector is called. "
//              << "global size is " << total_tdof << '\n';
//    Array<HYPRE_Int> fake_dof_offset(2);
//    fake_dof_offset[0] = 0;
//    fake_dof_offset[1] = total_tdof;
//    return (new HypreParVector(GetComm(), total_tdof, fake_dof_offset.GetData()));
// }
//version 2
// void ParGDSpace::AssembleProlongationMatrix(const Array<int> &els_id,
//                                             const DenseMatrix &local_mat)
// {
//    const Element *el = GetParMesh()->GetElement(0);
//    const FiniteElement *fe = fec->FiniteElementForGeometry(el->GetGeometryType());
//    const int num_dofs = fe->GetDof();
//    // need to transpose the local_mat
//    // local_mat.Transpose();
   
//    int nel = id.Size();
//    Array<int> el_dofs;
//    Array<int> col_index;
//    Array<int> row_index(num_dofs);
   
//    // Get the id of the element want to assemble in
//    int el_id = id[0];
//    GetElementVDofs(el_id, el_dofs);
//    cout << "Element dofs indices are: ";
//    el_dofs.Print(cout, el_dofs.Size());
//    cout << endl;
//    cout << "local mat size is " << local_mat.Height() << ' ' << local_mat.Width() << '\n';
//    col_index.SetSize(nel);
//    for(int e = 0; e < nel; e++)
//    {
//       col_index[e] = vdim * id[e];
//    }
//    for (int v = 0; v < vdim; v++)
//    {
//       el_dofs.GetSubArray(v * num_dofs, num_dofs, row_index);
//       cout << "local mat will be assembled into:\n";
//       cout << "rows: ";
//       row_index.Print(cout, row_index.Size());
//       cout << "columes: ";
//       col_index.Print(cout, col_index.Size());
//       cout << endl;
//       cP->SetSubMatrix(row_index, col_index, local_mat, 1);
//       row_index.LoseData();
//       // elements id also need to be shift accordingly
//       // col_index.SetSize(nel);
//       for (int e = 0; e < nel; e++)
//       {
//          col_index[e]++;
//       }
//    }
// }

} // end of namespace
