#include "mfem.hpp"
#include "thermal.hpp"
#include "gmi_egads.h"
#include "miso_egads.hpp"

#include <fstream>
#include <iostream>

using namespace std;
using namespace mfem;
using namespace miso;

#ifdef MFEM_USE_PUMI
int main(int argc, char *argv[])
{
#if 0
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
   const char *options_file = "mesh_move_options.json";
   args.AddOption(&options_file, "-o", "--options",
                  "Options file to use.");
   args.Parse();
   if (!args.Good())
   {
      args.PrintUsage(cout);
      return 1;
   }

   string opt_file_name(options_file);
   nlohmann::json options;
   nlohmann::json file_options;
   ifstream opts(opt_file_name);
   opts >> file_options;
   options.merge_patch(file_options);

   gmi_register_egads();
   gmi_egads_start();

   apf::Mesh2* pumi_mesh;
   std::unique_ptr<MeshType> mesh;
   apf::Mesh2* dummy_mesh;
   gmi_model* new_surf;
   string model_file_old = options["model-file-old"].template get<string>();
   string model_file_new = options["model-file-new"].template get<string>();
   string mesh_file = options["mesh"]["file"].template get<string>();
   string tess_file = options["tess-file-old"].template get<string>();
   
   cout << model_file_old << endl;
   PCU_Comm_Init();
#ifdef MFEM_USE_SIMMETRIX
    Sim_readLicenseFile(0);
    gmi_sim_start();
    gmi_register_sim();
#endif
   gmi_register_mesh();
   //new_surf = gmi_load(model_file_new.c_str());
   //pumi_mesh = apf::loadMdsMesh(model_file_old.c_str(), mesh_file.c_str()); 
   pumi_mesh = apf::loadMdsMesh(model_file_new.c_str(), mesh_file.c_str()); 
   //    pumi_mesh->verify();
   mesh.reset(new MeshType(MPI_COMM_WORLD, pumi_mesh));
   //begin surface mesh operationss
   apf::MeshEntity* bndn;
   apf::ModelEntity* bnd_face_old;
   apf::ModelEntity* bnd_face_new;
   apf::MeshIterator* itb = pumi_mesh->begin(0);
   apf::Vector3 param; apf::Vector3 param2;
   apf::Vector3 xold; apf::Vector3 xnew;
   mfem::Vector disp;
   disp.SetSize(3);
   Array<mfem::Vector> disp_list;
   //disp_list.SetSize(mesh->GetNV());
   gmi_egads_stop();

   cout << "Displacing" << endl;
   getBoundaryNodeDisplacement(model_file_old, model_file_new, tess_file,
                                pumi_mesh, &disp_list);

   cout << "Number of displaced nodes: " << disp_list.Size() << endl;                     

   for(int j = 1; j < disp_list.Size(); j++)
   {
      disp = disp_list[j];
      cout << "node: " << j << ", disp: " << disp(0) << " " << disp(1) << " " << disp(2) << endl;
   }
   // int j = 0;
   // while((bndn = pumi_mesh->iterate(itb)))
   // {
   //    pumi_mesh->getPoint(bndn, 0, xold);
   //    pumi_mesh->getParam(bndn, param);
   //    cout << "param point: " << param.x() << " "<<  param.y() << " " << param.z() << endl;

   //    bnd_face_old = pumi_mesh->toModel(bndn);
   //    int tag = pumi_mesh->getModelTag(bnd_face_old);
   //    cout << "tag: " << tag << endl;
   //    int type = pumi_mesh->getModelType(bnd_face_old);
   //    cout << "type: " << type << endl;
   //    //bnd_face_new = (apf::ModelEntity*)gmi_find(new_surf, type, tag);
   //    //gmi_reparam(new_surf, (gmi_ent*)bnd_face_old, &param[0], (gmi_ent*)bnd_face_new, &param2[0]);
   //    //gmi_eval(new_surf, (gmi_ent*)bnd_face_new, &param[0], &xnew[0]);
   //    gmi_eval(pumi_mesh->getModel(), (gmi_ent*)bnd_face_old, &param[0], &xnew[0]);
   //    disp(0) = xnew.x() - xold.x();
   //    disp(1) = xnew.y() - xold.y();
   //    disp(2) = xnew.z() - xold.z();
   //    disp_list[j] = disp;
   //    j++;
   //    cout << "node: " << j << ", oldx: " << xold.x() << " " << xold.y() << " " << xold.z() << endl;
   //    cout << "node: " << j << ", disp: " << disp(0) << " " << disp(1) << " " << disp(2) << endl;
   // }
   //gmi_egads_stop();




    PCU_Comm_Free();
#ifdef MFEM_USE_SIMMETRIX
    gmi_sim_stop();
    Sim_unregisterAllKeys();
#endif

   try
   {
    
   }
   catch (MISOException &exception)
   {

   }
#ifdef MFEM_USE_MPI
   MPI_Finalize();
#endif
#endif
}
#endif
