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
   const char *options_file = "mag_box_options.json";
   args.AddOption(&options_file, "-o", "--options",
                  "Options file to use.");
   int nxy = 2, nz = 2;
   args.AddOption(&nxy, "-nxy", "--numxy",
                  "Number of elements in x and y directions");
   args.AddOption(&nz, "-nz", "--numz",
                  "Number of elements in z direction");
   args.Parse();
   if (!args.Good())
   {
      args.PrintUsage(cout);
      return 1;
   }

   // generate a simple tet mesh
   std::unique_ptr<Mesh> mesh(new Mesh(nxy, nxy, nz,
                              Element::TETRAHEDRON, true /* gen. edges */, 1.0,
                              1.0, (double)nz / (double)nxy, true));

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


   // std::unique_ptr<mfem::Mesh> mesh(new Mesh("periodic_cube.mesh", 1, 1));
   // std::cout << "loaded mesh\n";
   // mesh->RemoveUnusedVertices();
   // std::cout << "removed unused\n";
   // mesh->RemoveInternalBoundaries();
   // std::cout << "removed internal boundaries\n";

   // ofstream mesh_ofs("periodic_cube_2.mesh");
   // mesh_ofs.precision(8);
   // mesh->Print(mesh_ofs);
   // mesh_ofs.close();

   // ofstream vtk_ofs("periodic_cube_2.vtk");
   // vtk_ofs.precision(8);
   // mesh->PrintVTK(vtk_ofs);
   // vtk_ofs.close();


   try
   {
      // construct the solver
      string opt_file_name(options_file);
      // MagnetostaticSolver solver(opt_file_name, move(mesh));
      auto solver = createSolver<MagnetostaticSolver>(opt_file_name, move(mesh));
      // unique_ptr<MagnetostaticSolver> solver(
      //    new MagnetostaticSolver(opt_file_name, nullptr));
      solver->solveForState();
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

