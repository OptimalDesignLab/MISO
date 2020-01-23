#include "mfem.hpp"
#include "magnetostatic.hpp"

#include <fstream>
#include <iostream>

using namespace std;
using namespace mfem;
using namespace mach;


int main(int argc, char *argv[])
{
   ostream *out;
#ifdef MFEM_USE_MPI
   // Initialize MPI if parallel
   MPI_Init(&argc, &argv);
   int rank;
   MPI_Comm_rank(MPI_COMM_WORLD, &rank);
   out = getOutStream(rank); 
#else
   out = getOutStream(0);
#endif

   // Parse command-line options
   OptionsParser args(argc, argv);
   const char *options_file = "magnetostatic_options.json";
   args.AddOption(&options_file, "-o", "--options",
                  "Options file to use.");
   int nx, ny, nz = 0;
   args.AddOption(&nx, "-nx", "--numx",
                  "Number of elements in x direction");
   args.AddOption(&ny, "-ny", "--numy",
                  "Number of elements in y direction");
   args.AddOption(&nz, "-nz", "--numz",
                  "Number of elements in z direction");
   args.Parse();
   if (!args.Good())
   {
      args.PrintUsage(cout);
      return 1;
   }

   // generate a simple tet mesh
   int num_edge = 20;
   std::unique_ptr<Mesh> mesh(new Mesh(nx, ny, nz,
                              Element::TETRAHEDRON, true /* gen. edges */, 1.0,
                              1.0, 1.0, true));

   mesh->ReorientTetMesh();

   // assign attributes to top and bottom sides
   for (int i = 0; i < mesh->GetNE(); ++i)
   {
      Element *elem = mesh->GetElement(i);

      Array<int> verts;
      elem->GetVertices(verts);

      bool below = true;
      for (int i = 0; i < 4; ++i)
      {
         auto vtx = mesh->GetVertex(verts[i]);
         if (vtx[1] <= 0.5)
         {
            below = below & true;
         }
         else
         {
            below = below & false;
         }
      }
      if (below)
      {
         elem->SetAttribute(1);
      }
      else
      {
         elem->SetAttribute(2);
      }
   }

   // ofstream mesh_ofs("test_cube.vtk");
   // mesh_ofs.precision(8);
   // mesh->PrintVTK(mesh_ofs);

   try
   {
      // construct the solver
      string opt_file_name(options_file);
      MagnetostaticSolver<3> solver(opt_file_name, move(mesh));
      // MagnetostaticSolver<3> solver(opt_file_name);
      // unique_ptr<MagnetostaticSolver<3>> solver(
      //    new MagnetostaticSolver<3>(opt_file_name, nullptr));
      solver.solveForState();
      // solver->solveForState();
      std::cout << "finish steady solve\n";
   }
   catch (MachException &exception)
   {
      exception.print_message();
   }
   catch (std::exception &exception)
   {
      cerr << exception.what() << endl;
   }
#ifdef MFEM_USE_MPI
   MPI_Finalize();
#endif
}

