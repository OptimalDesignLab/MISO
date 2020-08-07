#include <pybind11/pybind11.h>

#include <mpi4py/mpi4py.h>
#include "mpi4py_comm.hpp"

#include "mfem.hpp"

namespace py = pybind11;

using namespace mfem;

void initMesh(py::module &m)
{
   py::class_<mfem::ParMesh>(m, "Mesh")
      .def(py::init([](int nx, int ny, double sx, double sy, int order,
                       mpi4py_comm comm)
      {
         auto mesh = std::unique_ptr<Mesh>(new Mesh(nx, ny,
                                                    Element::TRIANGLE, true,
                                                    sx, sy, true));

         // mesh->SetCurvature(order, false, -1, 0);
         auto parmesh = std::unique_ptr<ParMesh>(new ParMesh(comm, *mesh));
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

      .def("Print", [](ParMesh &self, std::string &filename)
      {
         filename += ".mesh";
         std::ofstream ofs(filename);
         self.Print(ofs);
      }, 
      "Print the mesh in the default MFEM mesh format with the given filename "
      "(without an extension).",
      py::arg("filename"))

      .def("PrintVTU", [](ParMesh &self,
                          std::string &filename,
                          bool binary, 
                          bool high_order,
                          int compression_level)
      {
         VTKFormat format;
         if (binary)
         {
            format = VTKFormat::BINARY;
         }
         else
         {
            format = VTKFormat::ASCII;
         }
         self.PrintVTU(filename, format, high_order, compression_level);
      },
      "Print the mesh in VTU format with given filename "
      "(without an extension).",
      py::arg("filename"),
      py::arg("binary") = false,
      py::arg("high_order") = false,
      py::arg("compression_level") = 0)

      .def("getMeshSize", [](ParMesh &self)
      {
         self.EnsureNodes();
         return self.GetNodes()->FESpace()->GetVSize();
      }, "Return the number of nodes in the mesh")

      /// TODO: wrap GridFunType so I can return and access ordering
      // .def("getNodes", [](Mesh &self)
      // {
      //    self.EnsureNodes();
      //    Vector &nodes = *self.GetNodes();
      //    return nodes;
      // }, "Return the vector holding the coordinates of the mesh nodes")

      .def("getNodes", [](ParMesh &self, Vector &nodes)
      {
         self.EnsureNodes();
         nodes.MakeRef(*self.GetNodes(), 0, self.GetNodes()->FESpace()->GetVSize());
      }, "Set the vector holding the coordinates of the mesh nodes into the "
      "output argument `nodes`",
      py::arg("nodes"))

      .def("setNodes", [](ParMesh &self, Vector &nodes)
      {
         self.EnsureNodes();
         auto mesh_gf = self.GetNodes();
         mesh_gf->MakeRef(nodes, 0);
      }, "Set the coordinates of the mesh nodes to the new vector `nodes`",
      py::arg("nodes"))
   ;
}

// unique_ptr<Mesh> buildQuarterAnnulusMesh(int degree, int num_rad, int num_ang)
// {
//    auto mesh_ptr = unique_ptr<Mesh>(new Mesh(num_rad, num_ang,
//                                              Element::TRIANGLE, true /* gen. edges */,
//                                              2.0, M_PI*0.5, true));
//    // strategy:
//    // 1) generate a fes for Lagrange elements of desired degree
//    // 2) create a Grid Function using a VectorFunctionCoefficient
//    // 4) use mesh_ptr->NewNodes(nodes, true) to set the mesh nodes
   
//    // Problem: fes does not own fec, which is generated in this function's scope
//    // Solution: the grid function can own both the fec and fes
//    H1_FECollection *fec = new H1_FECollection(degree, 2 /* = dim */);
//    FiniteElementSpace *fes = new FiniteElementSpace(mesh_ptr.get(), fec, 2,
//                                                     Ordering::byVDIM);

//    // This lambda function transforms from (r,\theta) space to (x,y) space
//    auto xy_fun = [](const Vector& rt, Vector &xy)
//    {
//       xy(0) = (rt(0) + 1.0)*cos(rt(1)); // need + 1.0 to shift r away from origin
//       xy(1) = (rt(0) + 1.0)*sin(rt(1));
//    };
//    VectorFunctionCoefficient xy_coeff(2, xy_fun);
//    GridFunction *xy = new GridFunction(fes);
//    xy->MakeOwner(fec);
//    xy->ProjectCoefficient(xy_coeff);

//    mesh_ptr->NewNodes(*xy, true);
//    return mesh_ptr;
// }