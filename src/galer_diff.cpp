#include <fstream>
#include <iostream>
#include "default_options.hpp"
#include "galer_diff.hpp"
#include "sbp_fe.hpp"
using namespace std;
using namespace mach;
using namespace apf;
namespace mfem
{
GalerkinDifference::GalerkinDifference(const string &opt_file_name,
                                       Mesh2* mesh)
{
#ifndef MFEM_USE_PUMI
   mfem_error(" mfem needs to be build with pumi to use GalerkinDifference ")
#endif
   // should we keep this part to the problem specific file?
   nlohmann::json options = default_options;
   nlohmann::json file_options;
   ifstream options_file(opt_file_name);
   options_file >> file_options;
   options.merge_patch(file_options);
   cout << setw(3) << options << endl;
   
//    PCU_Comm_Init();
// #ifdef MFEM_USE_SIMMETRIX
//    Sim_readLicenseFile(0);
//    gmi_sim_start();
//    gmi_register_sim();
// #endif
//    gmi_register_mesh();
//    // load pumi mesh
//    pumi_mesh = loadMdsMesh(options["model-file"].get<string>().c_str(),
//                            options["pumi-mesh"]["file"].get<string>().c_str());
//    // verify pumi mesh
//    pumi_mesh->verify();
//    dim = pumi_mesh->getDimension();
//    nEle = pumi_mesh->count(dim);
// pmesh.reset(new MeshType(pumi_mesh, 1, 1));
   pumi_mesh = mesh;
   pmesh.reset(new MeshType(pumi_mesh, 1, 1));
   nEle = pmesh->GetNE();
   dim = pmesh->Dimension();
   cout << "check dim and nEle: " << dim << ' ' << nEle << '\n';
   // write meshx
   // ofstream sol_ofs("tri32_mfem.vtk");
   // sol_ofs.precision(14);
   // pmesh->PrintVTK(sol_ofs, 0);
   // apf::writeVtkFiles("pumi_mesh", pumi_mesh);
   // PCU_Comm_Free();

   // TODO:
   // 1. determine the size of cP. i.e. # of quadrature points and barycenters.
   // 2. call the mfem::FiniteElementSpace constructor.
   // 3. make sure that the dofs' order is consistent with that in bi/nonlinearforms

   // GD method requires DG fe collection
   // this function temporaly stay here
   degree = options["GD"]["degree"].get<int>();
   fec.reset(new DSBPCollection(options["space-dis"]["degree"].get<int>(),dim));
   Constructor(pmesh.get(), NULL, fec.get(), dim+2, Ordering::byVDIM);
   //Constructor(pmesh.get(), NULL, fec.get(), 1, Ordering::byVDIM);
   cout << "Galerkin Difference space is constructed.\n";
   cout << "Start to build the GD prolongation matrix of degree " << degree << '\n';
} // class constructor ends

// void GalerkinDifference::BuildNeighbourMat(DenseMatrix &nmat1, DenseMatrix &nmat2)
// {
//    // create pumi iterator over elements
//    pMeshIter it = pumi_mesh->begin(pumi_mesh_getDim(pumi_mesh));
//    pMeshEnt e;
//    Vector3 x;
//    // vector<int> nv1;
//    //cout << "pumi mesh element centers " << endl;
//    cout << "element neighbours " << endl;
//    int degree = 2;
//    int req_n = ((degree + 1) * (degree + 2)) / 2;
//    int max_n = 0;
//    int min_n = req_n;
//    // iterate over mesh elements to get maximum number of neighbours for an element.
//    // this provides the row size of neighbour matrices
//    while ((e = pumi_mesh->iterate(it)))
//    {
//       // create pumi mesh entity for neighbouring elements
//       Adjacent nels_e;
//       //get first neighbours (with shared edges)
//       getBridgeAdjacent(pumi_mesh, e, pumi_mesh_getDim(pumi_mesh) - 1,
//                         pumi_mesh_getDim(pumi_mesh), nels_e);
//       if (nels_e.size() > max_n)
//       {
//          max_n = nels_e.size();
//       }
//       if (nels_e.size() < min_n)
//       {
//          min_n = nels_e.size();
//       }
//    }
//    pumi_mesh->end(it); // end pumi iterations
//    cout << "max size " << endl;
//    cout << max_n << ", " << min_n << endl;
//    vector<int> nels;
//    //GetNeighbourSet(0, req_n, nels);
//    // set size of neighbour matrix
//    // To do: in 3D the # neighbours may be more than the required # neighbours
//    // nmat1.SetSize((max_n + req_n-min_n), nEle);
// }

// an overload function of previous one (more doable?)
void GalerkinDifference::BuildNeighbourMat(const mfem::Array<int> &elmt_id,
                                           mfem::DenseMatrix &mat_cent,
                                           mfem::DenseMatrix &mat_quad) const
{
   // resize the DenseMatrices and clean the data
   int num_el = elmt_id.Size();
   mat_cent.Clear(); 
   mat_cent.SetSize(dim, num_el);

   const FiniteElement *fe = fec->FiniteElementForGeometry(Geometry::TRIANGLE);
   const int num_dofs = fe->GetDof();
   // vectors that hold coordinates of quadrature points
   // used for duplication tests
   vector<double> quad_data;
   Vector quad_coord(dim); // used to store quadrature coordinate temperally
   ElementTransformation *eltransf;
   for(int j = 0; j < num_el; j++)
   {
      // Get and store the element center
      mfem::Vector cent_coord(dim);
      GetElementCenter(elmt_id[j], cent_coord);
      for(int i = 0; i < dim; i++)
      {
         mat_cent(i,j) = cent_coord(i);
      }
      
      // deal with quadrature points
      eltransf = pmesh->GetElementTransformation(elmt_id[j]);
      for(int k = 0; k < num_dofs; k++)
      {
         eltransf->Transform(fe->GetNodes().IntPoint(k), quad_coord);
         for(int di = 0; di < dim; di++)
         {
            quad_data.push_back(quad_coord(di));
         }
      }
   }
   // reset the quadrature point matrix
   mat_quad.Clear();
   int num_col = quad_data.size()/dim;
   mat_quad.SetSize(dim, num_col);
   for(int i = 0; i < num_col; i++)
   {
      for(int j = 0; j < dim; j++)
      {
         mat_quad(j,i) = quad_data[i*dim+j];
      }
   }
}

void GalerkinDifference::GetNeighbourSet(int id, int req_n,
                                    mfem::Array<int> &nels) const
{
   // this stores the elements for which we need neighbours
   vector<pMeshEnt> el;
   pMeshEnt e;
   // get pumi mesh entity (element) for the given id
   e = getMdsEntity(pumi_mesh, dim, id);
   // first, need to find neighbour of the given element
   el.push_back(e);
   // first entry in neighbour vector should be the element itself
   nels.LoseData(); // clean the queue vector 
   nels.Append(id);
   // iterate for finding element neighbours.
   // it stops when the # of elements in patch are equal/greater
   // than the minimum required # of elements in patch.
   while (nels.Size() < req_n)
   {
      // this stores the neighbour elements for which we need neighbours
      vector<pMeshEnt> elm;
      //get neighbours (with shared edges)
      for (int j = 0; j < el.size(); ++j)
      {
         // vector for storing neighbours of el[j]
         Adjacent nels_e1;
         // get neighbours
         getBridgeAdjacent(pumi_mesh, el[j], pumi_mesh_getDim(pumi_mesh) - 1,
                           pumi_mesh_getDim(pumi_mesh), nels_e1);
         // retrieve the id of neighbour elements
         // push in nels
         for (int i = 0; i < nels_e1.size(); ++i)
         {
            int nid;
            nid = getMdsIndex(pumi_mesh, nels_e1[i]);
            // check for element, push it if not there already
            if( -1 == nels.Find(nid))
            {
               nels.Append(nid);
            }
            // push neighbour elements for next iteration
            // and use them if required
            elm.push_back(nels_e1[i]);
         }
      }
      // resizing el to zero prevents finding neighbours of the same elements
      el.resize(0);
      // insert the neighbour elements in 'el' and iterate to find their neighbours if needed
      el.insert(end(el), begin(elm), end(elm));
   }
}

void GalerkinDifference::GetElementCenter(int id, mfem::Vector &cent) const
{
   cent.SetSize(pmesh->Dimension());
   int geom = pmesh->GetElement(id)->GetGeometryType();
   ElementTransformation *eltransf = pmesh->GetElementTransformation(id);
   eltransf->Transform(Geometries.GetCenter(geom), cent);
}

void GalerkinDifference::BuildGDProlongation() const
{
   const FiniteElement *fe = fec->FiniteElementForGeometry(Geometry::TRIANGLE);
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
   // loop over all the element:
   // 1. build the patch for each element,
   // 2. construct the local reconstruction operator
   // 3. assemble local reconstruction operator
   
   // vector that contains element id (resize to zero )
   mfem::Array<int> elmt_id;
   mfem::DenseMatrix cent_mat, quad_mat, local_mat;
   cout << "The size of the prolongation matrix is " << cP->Height() << " x " << cP->Width() << '\n';
   for (int i = 0; i < nEle; i++)
   {
      cout << "Element " << i << ":\n";
      // 1. construct the patch the patch
      // have more element than required to make it a underdetermined system
      GetNeighbourSet(i, nelmt, elmt_id);
      cout << "Elements id(s) in patch: ";
      elmt_id.Print(cout, elmt_id.Size());
      
      // 2. build the quadrature and barycenter coordinate matrices
      BuildNeighbourMat(elmt_id, cent_mat, quad_mat);
      // cout << "The element center matrix:\n";
      // cent_mat.Print(cout, cent_mat.Width());
      // cout << endl;
      // cout << "Quadrature points id matrix:\n";
      // quad_mat.Print(cout, quad_mat.Width());
      // cout << endl;

      // 3. buil the loacl reconstruction matrix
      buildInterpolation(dim, degree, num_dofs, cent_mat, quad_mat, local_mat);
      // cout << "Local reconstruction matrix R:\n";
      // local_mat.Print(cout, local_mat.Width());

      // 4. assemble them back to prolongation matrix
      AssembleProlongationMatrix(elmt_id, local_mat);
   }
   cP->Finalize();
   cout << "Check cP size: " << cP->Height() << " x " << cP->Width() << '\n';
   ofstream cp_save("cp_example.txt");
   cP->PrintMatlab(cp_save);
   cp_save.close();
}

// This function will be deleted because of the usage of dsbp
bool GalerkinDifference::duplicated(const Vector quad, const vector<double> data)
{
   bool duplicated;
   int data_size = data.size();
   MFEM_ASSERT(data_size % dim == 0," Quadrature data size is wrong.\n");
   for(int i = 0; i < data_size/dim; i++)
   {
      for(int di = 0; di < dim; di++)
      {
         if( quad(di) != data[i*dim+di] ){ return false; }
      }
   }
   // fall to pass the duplication test
   return true;
}

void GalerkinDifference::AssembleProlongationMatrix(const mfem::Array<int> &id,
                                            const DenseMatrix &local_mat) const
{
   // element id coresponds to the column indices
   // dofs id coresponds to the row indices
   // the local reconstruction matrix needs to be assembled `vdim` times
   const FiniteElement *fe = fec->FiniteElementForGeometry(Geometry::TRIANGLE);
   const int num_dofs = fe->GetDof();

   int nel = id.Size();
   Array<int> el_dofs;
   Array<int> col_index;
   Array<int> row_index(num_dofs);
   Array<Array<int>> dofs_mat(vdim);

   // Get the local basis for certain element
   // DenseMatrix el_mat(num_dofs, nel);
   // for (int c = 0; c < nel; c++)
   // {
   //    for (int r = 0; r < num_dofs; r++)
   //    {
   //       el_mat(r,c) = local_mat(r,c);
   //    }
   // }
   // cout << "Print the element mat:\n";
   // el_mat.Print(cout, nel);
   // Get the id of the element want to assemble in
   int el_id = id[0];
   GetElementVDofs(el_id, el_dofs);
   cout << "Element dofs indices are: ";
   el_dofs.Print(cout, el_dofs.Size());
   cout << endl;
   //cout << "local mat size is " << el_mat.Height() << ' ' << el_mat.Width() << '\n';
   col_index.SetSize(nel);
   for(int e = 0; e < nel; e++)
   {
      col_index[e] = vdim * id[e];
   }
   for (int v = 0; v < vdim; v++)
   {
      el_dofs.GetSubArray(v * num_dofs, num_dofs, row_index);
      cout << "local mat will be assembled into: ";
      row_index.Print(cout, num_dofs);
      cout << endl;
      cP->SetSubMatrix(row_index, col_index, local_mat, 1);
      row_index.LoseData();
      // elements id also need to be shift accordingly
      col_index.SetSize(nel);
      for (int e = 0; e < nel; e++)
      {
         col_index[e]++;
      }
   }
   

   // for(int i = 0; i < nel; i ++)
   // {
   //    GetElementVDofs(id[i], el_dofs);
   //    //cout << "The dofs size is " << el_dofs.Size() << " and data: ";
   //    for(int v = 0; v < vdim; v++)
   //    {
   //       el_dofs.GetSubArray(v * num_dofs, num_dofs, local_dofs);
   //       dofs_mat[v].Append(local_dofs);
   //       local_dofs.LoseData();
   //    }
   //    //el_dofs.Print(cout, el_dofs.Size());
   //    el_dofs.LoseData();
   // }
   // for(int v = 0; v < vdim; v++)
   // {
   //    cP->AddSubMatrix(dofs_mat[v], id, local_mat, 1);
   // }
}

} // namespace mfem
