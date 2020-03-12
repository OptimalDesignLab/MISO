#include "mach_egads.hpp"

extern "C" int EG_saveTess(egObject *tess, const char *name);
extern "C" int EG_loadTess(egObject *body, const char *name,
                           egObject **tess);

void getBoundaryNodeDisplacement(std::string something, apf::Mesh2* mesh, 

                                mfem::Array<mfem::Vector> disp_list)
{
    //start egads
    egObject* eg_context;
    int status = EG_open(&eg_context);

    apf::MeshEntity* bndn;
    apf::MeshIterator* itb = mesh->begin(0);
    apf::Vector3 xold;
    mfem::Vector disp(3);
    int global; //number of nodes
    int* ptype, *pindex;
    double* s_coords; //spatial coordinates 
    egObject* oldtess, *newtess;
    egObject* body;
    std::string newmodel = "../../sandbox/move_box_2.egads";
    std::string tessname = "../../sandbox/move_box.eto";

    //load model
    EG_loadModel(eg_context, 0, newmodel.c_str(), 
             &body);

    //get new tesselation, preserving mesh topology and getting new node points
    int loader = EG_loadTess(body, tessname.c_str(), &oldtess);
    int error = EG_mapTessBody(oldtess, body, &newtess);

    //check if successful
    if (error == 0)
    {
        //how will surface node ordering work?
        egTessel *btess = (egTessel *) newtess->blind;
        disp_list.SetSize(btess->nGlobal);
        //get new coordinates
        for(global = 0; global < btess->nGlobal; global++)
        {
            bndn = mesh->iterate(itb);
            mesh->getPoint(bndn, 0, xold);
            int error2 = EG_getGlobal(newtess, global, ptype, pindex, s_coords);
            if (error2 != 0)
            {
                //throw MachException("getNewBoundaryNodes()\n Failed to retrieve new coordinates at node "<< global <<"!");
            }
            disp(0) = s_coords[0] - xold.x();
            disp(1) = s_coords[1] - xold.y();
            disp(2) = s_coords[2] - xold.z();
            disp_list[global] = disp;
        }
    }
    else
    {
        //throw MachException("getNewBoundaryNodes()\n Remap of boundary nodes has failed!");
    }

    EG_close(eg_context);
}
