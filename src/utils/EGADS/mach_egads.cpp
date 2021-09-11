// #ifdef MFEM_USE_PUMI

#include "adept.h"
#include "mfem.hpp" // including mfem first is required or else there is a
#include "egads.h"  // compiler error

#include "mach_egads.hpp"

extern "C" int EG_saveTess(egObject *tess, const char *name);
extern "C" int EG_loadTess(egObject *body, const char *name,
                           egObject **tess);

using namespace mfem;

void mapSurfaceMesh(const std::string &old_model_file,
                    const std::string &new_model_file,
                    const std::string &tess_file,
                    HypreParVector &displacement)
   {
      // start egads
      ego eg_context;
      int status;
      status = EG_open(&eg_context);
      if (status != EGADS_SUCCESS)
         throw std::runtime_error("EG_open failed!\n");

      // load models
      ego old_model;
      status = EG_loadModel(eg_context, 0, old_model_file.c_str(), 
               &old_model);
      if (status != EGADS_SUCCESS)
         throw std::runtime_error("EG_loadModel failed!\n");

      ego new_model;
      status = EG_loadModel(eg_context, 0, new_model_file.c_str(), 
               &new_model);
      if (status != EGADS_SUCCESS)
         throw std::runtime_error("EG_loadModel failed!\n");

      // get bodies
      int oclass, mtype, nbody, *senses;
      ego old_geom, new_geom;
      ego *old_body;
      ego *new_body;

      status = EG_getTopology(old_model, &old_geom, 
                              &oclass, &mtype, NULL, &nbody, 
                              &old_body, &senses);
      if (status != EGADS_SUCCESS)
         throw std::runtime_error("EG_getTopology failed!\n");

      status = EG_getTopology(new_model, &new_geom, 
                              &oclass, &mtype, NULL, &nbody, 
                              &new_body, &senses);
      if (status != EGADS_SUCCESS)
         throw std::runtime_error("EG_getTopology failed!\n");

      ego old_tess;
      status = EG_loadTess(*old_body, tess_file.c_str(), &old_tess);
      if (status != EGADS_SUCCESS)
         throw std::runtime_error("EG_loadTess failed!\n");

      ego new_tess;
      status = EG_mapTessBody(old_tess, *new_body, &new_tess);
      if (status != EGADS_SUCCESS)
         throw std::runtime_error("EG_mapTessBody failed!\n");

      auto *old_raw_tess = static_cast<egTessel *>(old_tess->blind);
      // auto *raw_tess = static_cast<egTessel *>(new_tess->blind);

      int ptype, pindex;
      double xyz[3], xyz_old[3];
      // std::cout << "old_raw_tess->nGlobal: " << old_raw_tess->nGlobal << "\n";
      // std::cout << "raw_tess->nGlobal: " << raw_tess->nGlobal << "\n";

      bool two_dimensional = false;
      auto displacement_size = displacement.Size();
      if (displacement_size == 2*old_raw_tess->nGlobal )
         two_dimensional = true;

      // std::cout << "displacement_size: " << displacement_size << "\n";
      // std::cout << "tess size: " << old_raw_tess->nGlobal << "\n";
      // std::cout << "tess size (2): " << 2 * (old_raw_tess->nGlobal / 3) << "\n";
      // std::cout << "two dim: " << two_dimensional << "\n";

      for (int i = 1; i <= old_raw_tess->nGlobal; ++i)
      {
         EG_getGlobal(new_tess, i, &ptype, &pindex, xyz);
         EG_getGlobal(old_tess, i, &ptype, &pindex, xyz_old);
         if (two_dimensional)
         {
            displacement((i-1)*2 + 0) = xyz[0] - xyz_old[0];
            displacement((i-1)*2 + 1) = xyz[1] - xyz_old[1];
         }
         else
         {
            displacement((i-1)*3 + 0) = xyz[0] - xyz_old[0];
            displacement((i-1)*3 + 1) = xyz[1] - xyz_old[1];
            displacement((i-1)*3 + 2) = xyz[2] - xyz_old[2];
         }
         // std::cout << "(" << xyz[0] << ", " << xyz[1] << ", " << xyz[2] << ")\n";
      }




   }
   // , "Map an existing surface tessalation to a new body with the same topology",
   // py::arg("old_model"),
   // py::arg("new_model"),
   // py::arg("tess_file"),
   // py::arg("displacement"))
   // ;

// using namespace std;

// void getBoundaryNodeDisplacement(std::string oldmodel,
//                                  std::string newmodel, 
//                                  std::string tessname,
//                                  apf::Mesh2* mesh, 
//                                  mfem::Array<mfem::Vector> *disp_list)
// {
//    //start egads
//    egObject* eg_context;
//    int status = EG_open(&eg_context);

//    apf::MeshEntity* bndn;
//    apf::MeshIterator* itb = mesh->begin(0);
//    apf::Vector3 xold;
//    mfem::Vector disp(3);
//    int global; //number of nodes
//    int oclass, mtype, nbody, *senses;
//    int ptype, pindex;
//    double s_coords[3]; //spatial coordinates 
//    egObject* oldtess, *newtess;
//    egObject* geom1, * geom2, *model1, *model2;
//    egObject** body1, **body2;
//    // std::string newmodel = "../../sandbox/move_box_2.egads";
//    // std::string tessname = "../../sandbox/move_box.eto";

//    //load models
//    cout << "Loading Models" << endl;
//    int modelerr1 = EG_loadModel(eg_context, 0, oldmodel.c_str(), 
//             &model1);
//    int modelerr2 = EG_loadModel(eg_context, 0, newmodel.c_str(), 
//             &model2);

//    //get bodies
//    int topoerr1 = EG_getTopology(model1, &geom1, 
//                               &oclass, &mtype, NULL, &nbody, 
//                               &body1, &senses );
//    int topoerr2 = EG_getTopology(model2, &geom2, 
//                               &oclass, &mtype, NULL, &nbody, 
//                               &body2, &senses );

//    //get new tesselation, preserving mesh topology and getting new node points
//    cout << "Loading Tesselation" << endl;
//    int loader = EG_loadTess(*body1, tessname.c_str(), &oldtess);
//    cout << "Remapping Tesselation" << endl;
//    int error = EG_mapTessBody(oldtess, *body2, &newtess);
//    cout << "Status?: " << status << endl;                     
//    cout << "Model 1?: " << modelerr1 << endl;  
//    cout << "Model 2?: " << modelerr2 << endl;    
//    cout << "Body 1?: " << topoerr1 << endl;  
//    cout << "Body 2?: " << topoerr2 << endl;                  
//    cout << "Loader?: " << loader << endl;                     
//    cout << "Error?: " << error << endl; 

//    //check if successful
//    if (error == 0)
//    {
//       //how will surface node ordering work?
//       egTessel *btess = (egTessel *) oldtess->blind;
//       disp_list->SetSize(btess->nGlobal + 1);
//       cout << "Number of tess nodes: " << btess->nGlobal << endl;                     

//       //get new coordinates
//       for(global = 1; global <= btess->nGlobal; global++)
//       {
//             bndn = mesh->iterate(itb);
//             mesh->getPoint(bndn, 0, xold);
//             //cout << "Point: " << global << endl;                     
//             int error2 = EG_getGlobal(newtess, global, &ptype, &pindex, s_coords);
//             if (error2 != 0)
//             {
//                //throw MachException("getNewBoundaryNodes()\n Failed to retrieve new coordinates at node "<< global <<"!");
//             }
//             disp(0) = s_coords[0] - xold.x();
//             disp(1) = s_coords[1] - xold.y();
//             disp(2) = s_coords[2] - xold.z();
//             disp_list->GetData()[global] = disp;
//       }
//    }
//    else
//    {
//       //throw MachException("getNewBoundaryNodes()\n Remap of boundary nodes has failed!");
//    }

//    EG_close(eg_context);
// }

// apf::Mesh2* getNewMesh(std::string newmodel,
//                        std::string newmesh,
//                        mfem::Mesh* mfemmesh,
//                        apf::Mesh2* oldmesh)
// {
//    // PCU_Comm_Init();
//    //start egads
//    egObject* eg_context;
//    int status = EG_open(&eg_context);

//    apf::Mesh2* moved_mesh;
//    gmi_model* model;
//    //int modelerr = EG_loadModel(eg_context, 0, newmodel.c_str(), 
//    //         &model);
//    model = gmi_egads_load(newmodel.c_str());
//    moved_mesh = apf::createMdsMesh(model, oldmesh);
//    apf::MeshEntity* node;
//    apf::MeshIterator* it = moved_mesh->begin(0);
//    double* pointd;
//    apf::Vector3 point;
//    int global = 0;
//    while ((node = oldmesh->iterate(it)))
//    {
//       pointd = mfemmesh->GetVertex(global);
//       point.x() = pointd[0];
//       point.y() = pointd[1];
//       point.z() = pointd[2];
//       moved_mesh->setPoint(node, 0, point);
//       global++;
//    }
   
//    if(global != mfemmesh->GetNV() - 1)
//    {
//       cout << "Numbering Mismatch!" << endl;
//    }
   
//    moved_mesh->verify();
//    apf::writeMdsPart(moved_mesh, newmesh.c_str());
//    EG_close(eg_context);
//    PCU_Comm_Free();
//    return moved_mesh;
// }

// #endif
