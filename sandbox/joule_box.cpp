#include <fstream>
#include <iostream>

#include "mfem.hpp"
#include "miso.hpp"

// using namespace std;
// using namespace mfem;
// using namespace miso;

static double theta0;
static double t_final;
static double initialTemperature(const mfem::Vector &x);

int main(int argc, char *argv[])
{
   std::ostream *out;
#ifdef MFEM_USE_MPI
   // Initialize MPI if parallel
   MPI_Init(&argc, &argv);
   int rank;
   MPI_Comm_rank(MPI_COMM_WORLD, &rank);
   out = miso::getOutStream(rank); 
#else
   out = miso::getOutStream(0);
#endif

   // Parse command-line options
   mfem::OptionsParser args(argc, argv);
   const char *options_file = "joule_box_options.json";
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
      args.PrintUsage(*out);
      return 1;
   }

   std::string opt_file_name(options_file);
   nlohmann::json file_options;
   std::ifstream opts(opt_file_name);
   opts >> file_options;

   cout << setw(3) << file_options << "\n";
   file_options["problem-opts"]["init-temp"].get_to(theta0);
   file_options["thermal-opts"]["time-dis"]["t-final"].get_to(t_final);

   // generate a simple tet mesh
   std::unique_ptr<Mesh> mesh(new Mesh(nxy, nxy, nz,
                              Element::TETRAHEDRON, true /* gen. edges */, 1.0,
                              1.0, (double)nz / (double)nxy, true));


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

   try
   {
      // construct the solver
      miso::JouleSolver solver(opt_file_name, move(mesh));
      solver.initDerived();
      solver.setInitialCondition(initialTemperature);
      *out << "Solving..." << std::endl;
      solver.solveForState();
      *out << "Solving done." << std::endl;
      solver.printSolution("joule_out");
   }
   catch (miso::MISOException &exception)
   {
      exception.print_message();
   }
   catch (std::exception &exception)
   {
      std::cerr << exception.what() << std::endl;
   }
#ifdef MFEM_USE_MPI
   MPI_Finalize();
#endif
}

double initialTemperature(const mfem::Vector &x)
{
   // 70 deg Fahrenheit
   return theta0;
}
