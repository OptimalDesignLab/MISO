#include <fstream>
#include <iostream>
#include "default_options.hpp"
#include "galer_diff.hpp"
using namespace std;
using namespace mach;
using namespace apf;

namespace mfem
{

ParGDSpace::ParGDSpace(Mesh2 *apf_mesh, ParMesh *pm, const FiniteElementCollection *f,
                    	  int vdim, int ordering, int de, int p)
   : pr(p), SpaceType(pm, f, vdim, ordering)
{
   pumi_mesh = apf_mesh;
   dim = pm->Dimension();
   degree = de;
   local_tdof = vdim * GetParMesh()->GetNE();
   cout << "Before calculate element offsts.\n";
   GetParMesh()->ComputeGlobalElementOffset(); // something wrong here, for 1 processor probably works.
   cout << "After calculate element offsts.\n";
   MPI_Allreduce(&local_tdof, &total_tdof, 1, HYPRE_MPI_INT, MPI_SUM, GetComm());
   // dimension set up of the prolongation operator
   dof_start = (GetTrueDofOffsets())[0];
   dof_end = (GetTrueDofOffsets())[1] - 1;
   el_start = GetParMesh()->GetElementOffset();
   el_end = el_start + local_tdof - 1;

   BuildProlongationOperator();

   const char* p_save = "prolong";
   dynamic_cast<HypreParMatrix*>(P)->Print(p_save);
   if (pr == GetMyRank())
   {
      cout << "\n-------"
           << " ParGD Info "
           << "-------\n";
      cout << setw(15) << "Processor id:" << setw(8) << pr << '\n';
      cout << setw(15) << "vdim is:" << setw(8) << vdim << '\n';
      cout << setw(15) << "row start:" << setw(8) << dof_start << '\n';
      cout << setw(15) << "row end:" << setw(8) << dof_end << '\n';
      cout << setw(15) << "column start:" << setw(8) << el_start << '\n';
      cout << setw(15) << "column end:" << setw(8) << el_end << '\n';
      cout << setw(15) << "global  vdof:" << setw(8) << GlobalVSize() << '\n';
      cout << setw(15) << "global tdof:" << setw(8) << GlobalTrueVSize() << '\n';
      cout << setw(15) << "P size:" << setw(8) << dynamic_cast<HypreParMatrix*>(P)->Height() << " x "
        << dynamic_cast<HypreParMatrix*>(P)->Width() << '\n';
      cout << "--------------------------\n";
   }
}

// version 1
void ParGDSpace::BuildProlongationOperator()
{
   // initialize the hypre par matrix 
   HYPRE_IJMatrixCreate(GetComm(), dof_start, dof_end,
                        el_start, el_end, &ij_matrix);
   HYPRE_IJMatrixSetObjectType(ij_matrix, HYPRE_PARCSR);
   HYPRE_IJMatrixInitialize(ij_matrix);
   // The parallel prolongation matrix is built as a HypreParMatrix
   // follow the hypre API
   // assume the mesh only contains only 1 type of element
   const Element* el = GetParMesh()->GetElement(0);
   const FiniteElement *fe = fec->FiniteElementForGeometry(el->GetGeometryType());
   const int num_dofs = fe->GetDof();
   // allocate the space for the prolongation matrix
   // this step should be done in the constructor (probably)
   // should it be GetTrueVSize() ? or GetVSize()?
   // need a new method that directly construct a CSR format sparsematrix ？
   // cP = new mfem::SparseMatrix(GetVSize(), vdim * nEle);
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
   Array<int> elmt_id;
   mfem::DenseMatrix cent_mat, quad_mat, local_mat;
   //cout << "The size of the prolongation matrix is " << cP->Height() << " x " << cP->Width() << '\n';
   //int degree_actual;
   for (int i = 0; i < GetParMesh()->GetNE(); i++)
   {
      // 1. construct the patch the patch
      GetNeighbourSet(i, nelmt, elmt_id);
      cout << "Element id in patch " << i << ": ";
      elmt_id.Print(cout, elmt_id.Size());
      
      // 2. build the quadrature and barycenter coordinate matrices
      GetNeighbourMat(elmt_id, cent_mat, quad_mat);
      // cout << "The element center matrix:\n";
      // cent_mat.Print(cout, cent_mat.Width());
      // cout << endl;
      // cout << "Quadrature points id matrix:\n";
      // quad_mat.Print(cout, quad_mat.Width());
      // cout << endl;

      // 3. buil the loacl reconstruction matrix
      buildLSInterpolation(dim, degree, cent_mat, quad_mat, local_mat);
      // buildInterpolation(dim, degree, cent_mat, quad_mat, local_mat);
      // cout << "Local reconstruction matrix R:\n";
      // local_mat.Print(cout, local_mat.Width());
      // cout << "row-wise? " << *(local_mat.GetData()) << ' ' << *(local_mat.GetData()+1) << '\n'; 

      // 4. assemble them back to prolongation matrix
      AssembleProlongationMatrix(elmt_id, local_mat);
   }
   // finialize the hypre parmatrix
   HYPRE_IJMatrixAssemble(ij_matrix);
   HYPRE_IJMatrixGetObject(ij_matrix, (void**)&prolong);

   P = new HypreParMatrix((hypre_ParCSRMatrix*)(prolong), true);

   Vector diag(local_tdof);
   diag = 1.0;
   R = new SparseMatrix(diag);
}

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

void ParGDSpace::GetNeighbourSet(int id, int req_n, Array<int> &nels)
{
   /// this stores the elements for which we need neighbours
   vector<pMeshEnt> el;
   pMeshEnt e;
   /// get pumi mesh entity (element) for the given id
   e = apf::getMdsEntity(pumi_mesh, dim, id);
   /// first, need to find neighbour of the given element
   el.push_back(e);
   /// first entry in neighbour vector should be the element itself
   nels.SetSize(0); // clean the queue vector 
   nels.Append(id);
   /// iterate for finding element neighbours.
   /// it stops when the # of elements in patch are equal/greater
   /// than the minimum required # of elements in patch.
   while (nels.Size() < req_n)
   {
      /// this stores the neighbour elements for which we need neighbours
      vector<pMeshEnt> elm;
      ///get neighbours (with shared edges)
      for (int j = 0; j < el.size(); ++j)
      {
         /// vector for storing neighbours of el[j]
         Adjacent nels_e1;
         /// get neighbours
         getBridgeAdjacent(pumi_mesh, el[j], pumi_mesh_getDim(pumi_mesh) - 1,
                           pumi_mesh_getDim(pumi_mesh), nels_e1);
         /// retrieve the id of neighbour elements
         /// push in nels
         for (int i = 0; i < nels_e1.size(); ++i)
         {
            int nid;
            nid = getMdsIndex(pumi_mesh, nels_e1[i]);
            /// check for element, push it if not there already
            if (!(std::find(nels.begin(), nels.end(), nid) != nels.end()))
            {
               nels.Append(nid);
            }
            /// push neighbour elements for next iteration
            /// and use them if required
            elm.push_back(nels_e1[i]);
         }
      }
      /// resizing el to zero prevents finding neighbours of the same elements
      el.resize(0);
      /// insert the neighbour elements in 'el' and iterate to find their neighbours if needed
      el.insert(end(el), begin(elm), end(elm));
   }
}

void ParGDSpace::GetNeighbourMat(Array<int> &els_id, DenseMatrix &mat_cent,
                                DenseMatrix &mat_quad) const
{
   // assume the mesh only contains only 1 type of element
   const Element* el = GetParMesh()->GetElement(0);
   const FiniteElement *fe = fec->FiniteElementForGeometry(el->GetGeometryType());
   const int num_dofs = fe->GetDof();

   // resize the DenseMatrices and clean the data
   int num_el = els_id.Size();
   mat_cent.Clear(); 
   mat_cent.SetSize(dim, num_el);
   mat_quad.Clear();
   mat_quad.SetSize(dim, num_dofs);
   // vectors that hold coordinates of quadrature points
   // used for duplication tests
   vector<double> quad_data;
   Vector quad_coord(dim); // used to store quadrature coordinate temperally
   Vector cent_coord(dim);
   for(int j = 0; j < num_el; j++)
   {
      // Get and store the element center
      GetElementCenter(els_id[j], cent_coord);
      for(int i = 0; i < dim; i++)
      {
         mat_cent(i,j) = cent_coord(i);
      }
      
      // deal with quadrature points
      // following commented on Apr 19 2021
      // eltransf = mesh->GetElementTransformation(els_id[j]);
      // for(int k = 0; k < num_dofs; k++)
      // {
      //    eltransf->Transform(fe->GetNodes().IntPoint(k), quad_coord);
      //    for(int di = 0; di < dim; di++)
      //    {
      //       quad_data.push_back(quad_coord(di));
      //    }
      // }
   }

   // deal with quad points
   ElementTransformation *eltransf = GetParMesh()->GetElementTransformation(els_id[0]);
   for (int j = 0; j < num_dofs; j++)
   {
      eltransf->Transform(fe->GetNodes().IntPoint(j), quad_coord);
      for (int i = 0; i < dim; i++)
      {
         mat_quad(i, j) = quad_coord(i);
      }
   }

   // reset the quadrature point matrix
   // following comment on Apr 19 2021
   // mat_quad.Clear();
   // int num_col = quad_data.size()/dim;
   // mat_quad.SetSize(dim, num_col);
   // for(int i = 0; i < num_col; i++)
   // {
   //    for(int j = 0; j < dim; j++)
   //    {
   //       mat_quad(j,i) = quad_data[i*dim+j];
   //    }
   // }
}

void ParGDSpace::GetElementCenter(int id, Vector &cent) const
{
   cent.SetSize(dim);
   int geom = mesh->GetElement(id)->GetGeometryType();
   ElementTransformation *eltransf = mesh->GetElementTransformation(id);
   eltransf->Transform(Geometries.GetCenter(geom), cent);
}

// version 1
void ParGDSpace::AssembleProlongationMatrix(const Array<int> &els_id,
                                            DenseMatrix &local_mat)
{
   const Element *el = GetParMesh()->GetElement(0);
   const FiniteElement *fe = fec->FiniteElementForGeometry(el->GetGeometryType());
   const int num_dofs = fe->GetDof();
   // need to transpose the local_mat given the data format
   local_mat.Transpose();
   
   int nel = els_id.Size();
   Array<int> el_dofs;
   Array<int> col_index(nel*num_dofs);
   Array<int> row_index(num_dofs);
 
   int el_id = els_id[0];
   GetElementVDofs(el_id, el_dofs);
   Array<int> ncols(num_dofs);
   ncols = nel;
   // cout << "ncols is: "; ncols.Print(cout, ncols.Size());
   // cout << "mat size is " << local_mat.Height() << " x " << local_mat.Width() << '\n';
   // cout << "col_index size is " << col_index.Size() << '\n';
   for (int d = 0; d < num_dofs; d++)
   {
      for (int e = 0; e < nel; e++)
      {
         col_index[d * nel + e] = vdim * els_id[e];
      }
   }
   for (int v = 0; v < vdim; v++)
   {
      el_dofs.GetSubArray(v * num_dofs, num_dofs, row_index);
      cout << '\t' <<  v << " assemble location:\n";
      cout << "\t\tRow:"; row_index.Print(cout, row_index.Size());
      cout << "\t\tCol:"; col_index.Print(cout, col_index.Size());
      HYPRE_IJMatrixSetValues(ij_matrix, num_dofs, ncols.GetData(), row_index.GetData(),
                             col_index.GetData(), local_mat.GetData());
      row_index.LoseData();
      for (int e = 0; e < nel * num_dofs; e++)
      {
         col_index[e]++;
      }
   }
}

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
