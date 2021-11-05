#include <pybind11/pybind11.h>

#include <mpi4py/mpi4py.h>
#include "mpi4py_comm.hpp"

#include "mfem.hpp"

#ifdef MFEM_USE_PUMI

#include "apfMDS.h"
#include "PCU.h"
#include "apfConvert.h"
#include "crv.h"
#include "gmi_mesh.h"
#include "gmi_null.h"

#ifdef MFEM_USE_SIMMETRIX
#include "SimUtil.h"
#include "gmi_sim.h"
#endif  // MFEM_USE_SIMMETRIX

#ifdef MFEM_USE_EGADS
#include "gmi_egads.h"
#endif  // MFEM_USE_EGADS

namespace
{
/// function to figure out if tet element is next to a model surface
bool isBoundaryTet(apf::Mesh2 *m, apf::MeshEntity *e)
{
   apf::MeshEntity *dfs[12];
   int nfs = m->getDownward(e, 2, dfs);
   for (int i = 0; i < nfs; i++)
   {
      int mtype = m->getModelType(m->toModel(dfs[i]));
      if (mtype == 2)
      {
         return true;
      }
   }
   return false;
}
}  // anonymous namespace

#endif  // MFEM_USE_PUMI

namespace py = pybind11;

using namespace mfem;

void initMesh(py::module &m)
{
   py::class_<mfem::ParMesh>(m, "Mesh")
       .def(py::init(
                [](int nx,
                   int ny,
                   double sx,
                   double sy,
                   int order,
                   mpi4py_comm comm)
                {
                   auto mesh = Mesh::MakeCartesian2D(
                       nx, ny, Element::TRIANGLE, true, sx, sy);

                   // mesh->SetCurvature(order, false, -1, 0);
                   auto parmesh =
                       std::make_unique<ParMesh>(comm, mesh);
                   parmesh->SetCurvature(order, false, -1, 1);
                   return parmesh;
                }),
            "Creates mesh of the given order for the rectangle [0,sx]x[0,sy], "
            "divided into 2*nx*ny triangles",
            py::arg("nx"),
            py::arg("ny"),
            py::arg("sx"),
            py::arg("sy"),
            py::arg("order"),
            py::arg("comm") = mpi4py_comm(MPI_COMM_WORLD))

       .def(
           py::init(
               [](int nx,
                  int ny,
                  int nz,
                  double sx,
                  double sy,
                  double sz,
                  mpi4py_comm comm)
               {
                  auto mesh = Mesh::MakeCartesian3D(
                      nx, ny, nz, Element::TETRAHEDRON, sx, sy, sz, true);

                  auto parmesh =
                      std::make_unique<ParMesh>(comm, mesh);
                  return parmesh;
               }),
           "Creates mesh for the parallelepiped [0,1]x[0,1]x[0,1], divided into"
           "6 x `nx` x `ny` x 'nz' tetrahedrons",
           py::arg("nx"),
           py::arg("ny"),
           py::arg("nz"),
           py::arg("sx"),
           py::arg("sy"),
           py::arg("sz"),
           py::arg("comm") = mpi4py_comm(MPI_COMM_WORLD))

#ifdef MFEM_USE_PUMI
       .def(
           py::init(
               [](const std::string &model_file,
                  const std::string &mesh_file,
                  mpi4py_comm comm)
               {
   // PCU_Comm_Init();

#ifdef MFEM_USE_SIMMETRIX
                  Sim_readLicenseFile(0);
                  gmi_sim_start();
                  gmi_register_sim();
#endif
#ifdef MFEM_USE_EGADS
                  gmi_register_egads();
                  gmi_egads_start();
#endif
                  gmi_register_mesh();
                  auto *pumi_mesh =
                      apf::loadMdsMesh(model_file.c_str(), mesh_file.c_str());

                  // int mesh_dim = pumi_mesh->getDimension();
                  // int nEle = pumi_mesh->count(mesh_dim);
                  // int ref_levels = (int)floor(log(10000. / nEle) / log(2.) /
                  // mesh_dim); Perform Uniform refinement if (ref_levels > 1)
                  // {
                  //    ma::Input* uniInput =
                  //    ma::configureUniformRefine(pumi_mesh, ref_levels);
                  //    ma::adapt(uniInput);
                  // }

                  /// TODO: change this to use argument
                  /// If it is higher order change shape
                  // if (order > 1)
                  // {
                  //     crv::BezierCurver bc(pumi_mesh, order, 2);
                  //     bc.run();
                  // }

                  pumi_mesh->verify();

                  apf::Numbering *aux_num = apf::createNumbering(
                      pumi_mesh, "aux_numbering", pumi_mesh->getShape(), 1);

                  apf::MeshIterator *it = pumi_mesh->begin(0);
                  apf::MeshEntity *v = nullptr;
                  int count = 0;
                  while ((v = pumi_mesh->iterate(it)) != nullptr)
                  {
                     apf::number(aux_num, v, 0, 0, count++);
                  }
                  pumi_mesh->end(it);

                  auto *mesh = new ParPumiMesh(comm, pumi_mesh);

                  it = pumi_mesh->begin(pumi_mesh->getDimension());
                  count = 0;
                  while ((v = pumi_mesh->iterate(it)) != nullptr)
                  {
                     if (count > 10)
                     {
                        break;
                     }
                     printf("at element %d =========\n", count);
                     if (isBoundaryTet(pumi_mesh, v))
                     {
                        printf("tet is connected to the boundary\n");
                     }
                     else
                     {
                        printf("tet is NOT connected to the boundary\n");
                     }
                     apf::MeshEntity *dvs[12];
                     int nd = pumi_mesh->getDownward(v, 0, dvs);
                     for (int i = 0; i < nd; i++)
                     {
                        int id = apf::getNumber(aux_num, dvs[i], 0, 0);
                        printf("%d ", id);
                     }
                     printf("\n");
                     Array<int> mfem_vs;
                     mesh->GetElementVertices(count, mfem_vs);
                     for (int i = 0; i < mfem_vs.Size(); i++)
                     {
                        printf("%d ", mfem_vs[i]);
                     }
                     printf("\n");
                     printf("=========\n");
                     count++;
                  }

                  /// Add attributes based on reverse classification
                  // Boundary faces
                  int dim = mesh->Dimension();
                  apf::MeshIterator *itr = pumi_mesh->begin(dim - 1);
                  apf::MeshEntity *ent = nullptr;
                  int ent_cnt = 0;
                  while ((ent = pumi_mesh->iterate(itr)) != nullptr)
                  {
                     apf::ModelEntity *me = pumi_mesh->toModel(ent);
                     if (pumi_mesh->getModelType(me) == (dim - 1))
                     {
                        // Get tag from model by  reverse classification
                        int tag = pumi_mesh->getModelTag(me);
                        (mesh->GetBdrElement(ent_cnt))->SetAttribute(tag);
                        ent_cnt++;
                     }
                  }
                  pumi_mesh->end(itr);

                  // Volume faces
                  itr = pumi_mesh->begin(dim);
                  ent_cnt = 0;
                  while ((ent = pumi_mesh->iterate(itr)) != nullptr)
                  {
                     apf::ModelEntity *me = pumi_mesh->toModel(ent);
                     int tag = pumi_mesh->getModelTag(me);
                     mesh->SetAttribute(ent_cnt, tag);
                     ent_cnt++;
                  }
                  pumi_mesh->end(itr);

                  // Apply the attributes
                  mesh->SetAttributes();

                  pumi_mesh->destroyNative();
                  apf::destroyMesh(pumi_mesh);
      // PCU_Comm_Free();
#ifdef MFEM_USE_SIMMETRIX
                  gmi_sim_stop();
                  Sim_unregisterAllKeys();
#endif  // MFEM_USE_SIMMETRIX

#ifdef MFEM_USE_EGADS
                  gmi_egads_stop();
#endif  // MFEM_USE_EGADS

                  return mesh;
                  // #else
                  //          throw MachException("mfem::ParMesh::init()\n"
                  //                              "\tMFEM was not built with
                  //                              PUMI!\n"
                  //                              "\trecompile MFEM with
                  //                              PUMI\n");
               }),
           "Loads a PUMI mesh from an .smb file and associated model file, and "
           "converts to an MFEM mesh",
           py::arg("model_file"),
           py::arg("mesh_file"),
           py::arg("comm") = mpi4py_comm(MPI_COMM_WORLD))
#endif  // MFEM_USE_PUMI

       .def(
           "Print",
           [](ParMesh &self, std::string &filename)
           {
              filename += ".mesh";
              std::ofstream ofs(filename);
              self.Print(ofs);
           },
           "Print the mesh in the default MFEM mesh format with the given "
           "filename "
           "(without an extension).",
           py::arg("filename"))

       .def(
           "PrintVTU",
           [](ParMesh &self,
              std::string &filename,
              bool binary,
              bool high_order,
              int compression_level)
           {
              VTKFormat format = [&]() {
               if (binary)
               {
                  return VTKFormat::BINARY;
               }
               else
               {
                  return VTKFormat::ASCII;
               }
              }();
              self.PrintVTU(filename, format, high_order, compression_level);
           },
           "Print the mesh in VTU format with given filename "
           "(without an extension).",
           py::arg("filename"),
           py::arg("binary") = false,
           py::arg("high_order") = false,
           py::arg("compression_level") = 0)

       .def(
           "getMeshSize",
           [](ParMesh &self)
           {
              self.EnsureNodes();
              return self.GetNodes()->FESpace()->GetVSize();
           },
           "Return the number of nodes in the mesh")

       /// TODO: wrap GridFunType so I can return and access ordering
       // .def("getNodes", [](Mesh &self)
       // {
       //    self.EnsureNodes();
       //    Vector &nodes = *self.GetNodes();
       //    return nodes;
       // }, "Return the vector holding the coordinates of the mesh nodes")

       .def(
           "getNodes",
           [](ParMesh &self, Vector &nodes)
           {
              self.EnsureNodes();
              nodes.MakeRef(
                  *self.GetNodes(), 0, self.GetNodes()->FESpace()->GetVSize());
           },
           "Set the vector holding the coordinates of the mesh nodes into the "
           "output argument `nodes`",
           py::arg("nodes"))

       .def(
           "setNodes",
           [](ParMesh &self, Vector &nodes)
           {
              self.EnsureNodes();
              auto *mesh_gf = self.GetNodes();
              mesh_gf->MakeRef(nodes, 0);
           },
           "Set the coordinates of the mesh nodes to the new vector `nodes`",
           py::arg("nodes"))

       .def(
           "addDisplacement",
           [](ParMesh &self, ParGridFunction &displacement)
           {
              self.EnsureNodes();
              auto &mesh_coords = *self.GetNodes();
              mesh_coords += displacement;
           },
           "Displace the coordinates of the mesh nodes by the values in the "
           "field `displacement`",
           py::arg("displacement"));
}

// unique_ptr<Mesh> buildQuarterAnnulusMesh(int degree, int num_rad, int
// num_ang)
// {
//    auto mesh_ptr = unique_ptr<Mesh>(new Mesh(num_rad, num_ang,
//                                              Element::TRIANGLE, true /* gen.
//                                              edges */, 2.0, M_PI*0.5, true));
//    // strategy:
//    // 1) generate a fes for Lagrange elements of desired degree
//    // 2) create a Grid Function using a VectorFunctionCoefficient
//    // 4) use mesh_ptr->NewNodes(nodes, true) to set the mesh nodes

//    // Problem: fes does not own fec, which is generated in this function's
//    scope
//    // Solution: the grid function can own both the fec and fes
//    H1_FECollection *fec = new H1_FECollection(degree, 2 /* = dim */);
//    FiniteElementSpace *fes = new FiniteElementSpace(mesh_ptr.get(), fec, 2,
//                                                     Ordering::byVDIM);

//    // This lambda function transforms from (r,\theta) space to (x,y) space
//    auto xy_fun = [](const Vector& rt, Vector &xy)
//    {
//       xy(0) = (rt(0) + 1.0)*cos(rt(1)); // need + 1.0 to shift r away from
//       origin xy(1) = (rt(0) + 1.0)*sin(rt(1));
//    };
//    VectorFunctionCoefficient xy_coeff(2, xy_fun);
//    GridFunction *xy = new GridFunction(fes);
//    xy->MakeOwner(fec);
//    xy->ProjectCoefficient(xy_coeff);

//    mesh_ptr->NewNodes(*xy, true);
//    return mesh_ptr;
// }