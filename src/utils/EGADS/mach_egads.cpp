#include "mach_egads.hpp"
#include "egads.h"
#include "adept.h"
#include "mach_egads.hpp"

extern "C" int EG_saveTess(egObject *tess, const char *name);
extern "C" int EG_loadTess(egObject *body, const char *name,
                           egObject **tess);

using namespace std;

void getBoundaryNodeDisplacement(std::string oldmodel,
                                std::string newmodel, 
                                std::string tessname,
                                apf::Mesh2* mesh, 
                                mfem::Array<mfem::Vector> *disp_list)
{
    //start egads
    egObject* eg_context;
    int status = EG_open(&eg_context);

    apf::MeshEntity* bndn;
    apf::MeshIterator* itb = mesh->begin(0);
    apf::Vector3 xold;
    mfem::Vector disp(3);
    int global; //number of nodes
    int oclass, mtype, nbody, *senses;
    int ptype, pindex;
    double s_coords[3]; //spatial coordinates 
    egObject* oldtess, *newtess;
    egObject* geom1, * geom2, *model1, *model2;
    egObject** body1, **body2;
    // std::string newmodel = "../../sandbox/move_box_2.egads";
    // std::string tessname = "../../sandbox/move_box.eto";

    //load models
    cout << "Loading Models" << endl;
    int modelerr1 = EG_loadModel(eg_context, 0, oldmodel.c_str(), 
             &model1);
    int modelerr2 = EG_loadModel(eg_context, 0, newmodel.c_str(), 
             &model2);

    //get bodies
    int topoerr1 = EG_getTopology(model1, &geom1, 
                              &oclass, &mtype, NULL, &nbody, 
                              &body1, &senses );
    int topoerr2 = EG_getTopology(model2, &geom2, 
                              &oclass, &mtype, NULL, &nbody, 
                              &body2, &senses );

    //get new tesselation, preserving mesh topology and getting new node points
    cout << "Loading Tesselation" << endl;
    int loader = EG_loadTess(*body1, tessname.c_str(), &oldtess);
    cout << "Remapping Tesselation" << endl;
    int error = EG_mapTessBody(oldtess, *body2, &newtess);
    cout << "Status?: " << status << endl;                     
    cout << "Model 1?: " << modelerr1 << endl;  
    cout << "Model 2?: " << modelerr2 << endl;    
    cout << "Body 1?: " << topoerr1 << endl;  
    cout << "Body 2?: " << topoerr2 << endl;                  
    cout << "Loader?: " << loader << endl;                     
    cout << "Error?: " << error << endl; 

    //check if successful
    if (error == 0)
    {
        //how will surface node ordering work?
        egTessel *btess = (egTessel *) newtess->blind;
        disp_list->SetSize(41);
        cout << "Number of tess nodes: " << btess->nGlobal << endl;                     

        //get new coordinates
        for(global = 1; global <= 40; global++)
        {
            bndn = mesh->iterate(itb);
            mesh->getPoint(bndn, 0, xold);
            cout << "Point: " << global << endl;                     
            int error2 = EG_getGlobal(newtess, global, &ptype, &pindex, s_coords);
            if (error2 != 0)
            {
                //throw MachException("getNewBoundaryNodes()\n Failed to retrieve new coordinates at node "<< global <<"!");
            }
            disp(0) = s_coords[0] - xold.x();
            disp(1) = s_coords[1] - xold.y();
            disp(2) = s_coords[2] - xold.z();
            disp_list->GetData()[global] = disp;
        }
    }
    else
    {
        //throw MachException("getNewBoundaryNodes()\n Remap of boundary nodes has failed!");
    }

    EG_close(eg_context);
}