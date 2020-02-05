/// Solve the steady isentropic vortex problem on a quarter annulus
#include<random>
#include "adept.h"

#include "mfem.hpp"
#include "euler.hpp"
#include "galer_diff.hpp"
#include <fstream>
#include <iostream>

using namespace std;
using namespace mfem;
using namespace mach;
using namespace apf;

std::default_random_engine gen(std::random_device{}());
std::uniform_real_distribution<double> normal_rand(-1.0,1.0);

// This function will be used to check the local R and the assembled prolongation matrix
int main(int argc, char *argv[])
{
   // the specfic option file for this problem
   const char *options_file = "galerkin_difference.json";
   nlohmann::json options;
   ifstream option_source(options_file);
   option_source >> options;
   option_source.close();

   // write the petsc option file
#ifdef MFEM_USE_PETSC
   const char *petscrc_file = "galerkin_difference.petsc";
   ofstream petscoptions(petscrc_file);
   const string linearsolver_name = options["petscsolver"]["ksptype"].get<string>();
   const string prec_name = options["petscsolver"]["pctype"].get<string>();
   petscoptions << "-solver_ksp_type " << linearsolver_name << '\n';
   petscoptions << "-prec_pc_type " << prec_name << '\n';
   petscoptions.close();
#endif

#ifdef MFEM_USE_MPI
   // Initialize MPI if parallel
   int num_procs, myid;
   MPI_Init(&argc, &argv);
   MPI_Comm_size(MPI_COMM_WORLD, &num_procs);
   MPI_Comm_rank(MPI_COMM_WORLD, &myid);
#else
   int myid = 0;
#endif
#ifdef MFEM_USE_PETSC
   MFEMInitializePetsc(NULL, NULL, petscrc_file, NULL);
#endif
   int nx = 1, ny = 1;
   int degree = 2;
   int dim; // space dimension of the mesh
   int nEle; // number of element in the pumi mesh
   // Parse command-line options
   OptionsParser args(argc, argv);
   args.AddOption(&options_file, "-o", "--options",
                  "Options file to use.");
   args.AddOption(&degree, "-d", "--degree", "poly. degree of mesh mapping");
   args.AddOption(&nx, "-nr", "--num-rad", "number of radial segments");
   args.AddOption(&ny, "-nt", "--num-thetat", "number of angular segments");
   args.Parse();
   if (!args.Good())
   {
      args.PrintUsage(cout);
      return 1;
   }
  
   try
   {
      PCU_Comm_Init();
#ifdef MFEM_USE_SIMMETRIX
      Sim_readLicenseFile(0);
      gmi_sim_start();
      gmi_register_sim();
#endif
      gmi_register_mesh();

      apf::Mesh2 *pumi_mesh;
      pumi_mesh = apf::loadMdsMesh(options["model-file"].get<string>().c_str(),
                           options["pumi-mesh"]["file"].get<string>().c_str());
      pumi_mesh->verify();

      dim = pumi_mesh->getDimension();
      nEle = pumi_mesh->count(dim);
      cout << "Mesh dimension is " << dim << '\n';
      cout << "Number of element " << nEle << '\n';
      
      
      cout << "Construct the GD fespace.\n";
      GalerkinDifference gd(options_file, pumi_mesh);
      gd.BuildGDProlongation();

      // test the prolongation 
      mfem::Vector cent_val(4*nEle);
      mfem::Vector quad_val(gd.GetVSize());
      cout << "Size of cent_val " << cent_val.Size() << '\n';
      cout << "Size of quad_val " << quad_val.Size() << '\n';

      cent_val = 1.0;
      gd.GetProlongationMatrix()->Mult(cent_val, quad_val);
      ofstream 
      cout << "Check the result:\n";
      quad_val.Print(cout, 4);
      PCU_Comm_Free();
   }


   catch (MachException &exception)
   {
      exception.print_message();
   }
   catch (std::exception &exception)
   {
      cerr << exception.what() << endl;
   }

#ifdef MFEM_USE_PETSC
   MFEMFinalizePetsc();
#endif
#ifdef MFEM_USE_MPI
   MPI_Finalize();
#endif
}

