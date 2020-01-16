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
   args.Parse();
   if (!args.Good())
   {
      args.PrintUsage(cout);
      return 1;
   }

   // generate a simple tet mesh
   int num_edge = 20;
   std::unique_ptr<Mesh> mesh(new Mesh(options["num-elem"]["x-num"].get<int>(), 
                              options["num-elem"]["y-num"].get<int>(),
                              options["num-elem"]["z-num"].get<int>(),
                              Element::TETRAHEDRON, true /* gen. edges */, 1.0,
                              1.0, 0.1, true));

   mesh->ReorientTetMesh();

   // // assign attributes to top and bottom sides
   // for (int i = 0; i < mesh->GetNE(); ++i)
   // {
   //    Element *elem = mesh->GetElement(i);

   //    Array<int> verts;
   //    elem->GetVertices(verts);

   //    bool below = true;
   //    for (int i = 0; i < 4; ++i)
   //    {
   //       auto vtx = mesh->GetVertex(verts[i]);
   //       if (vtx[1] <= 0.5)
   //       {
   //          below = below & true;
   //       }
   //       else
   //       {
   //          below = below & false;
   //       }
   //    }
   //    if (below)
   //    {
   //       elem->SetAttribute(1);
   //    }
   //    else
   //    {
   //       elem->SetAttribute(2);
   //    }
   // }

   // ofstream mesh_ofs("test_cube.vtk");
   // mesh_ofs.precision(8);
   // mesh->PrintVTK(mesh_ofs);

   try
   {
      // construct the solver
      string opt_file_name(options_file);
      // MagnetostaticSolver solver(opt_file_name, move(mesh));
      MagnetostaticSolver solver(opt_file_name);
      solver.solveForState();
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

