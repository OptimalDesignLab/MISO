// #ifdef MFEM_USE_PUMI

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

#include "mfem.hpp"

#include "apfMDS.h"
#include "PCU.h"
#include "apfConvert.h"
#include "crv.h"
#include "gmi_mesh.h"

// #ifdef MFEM_USE_EGADS
#include "egads.h"
#include "gmi_egads.h"

extern "C" int EG_saveTess(egObject *tess, const char *name);
extern "C" int EG_loadTess(egObject *body, const char *name,
                           egObject **tess);

// #endif // MFEM_USE_EGADS

namespace py = pybind11;

using namespace mfem;

void initMeshMotion(py::module &m)
{
   m.def("mapSurfaceMesh", [](const std::string &old_model_file,
                              const std::string &new_model_file,
                              const std::string &tess_file,
                              py::array_t<double> buffer)
   {
      std::cout << "\n\ncalling mapSurfaceMesh!\n\n\n";
      /* Request a buffer descriptor from Python */
      py::buffer_info info = buffer.request();

      /* Some sanity checks ... */
      if (info.format != py::format_descriptor<double>::format())
         throw std::runtime_error("Incompatible format:\n"
                                    "\texpected a double array!");

      if (info.ndim != 1)
         throw std::runtime_error("Incompatible dimensions:\n"
                                    "\texpected a 1D array!");

      auto new_coords_size = info.shape[0];
      
      auto new_coords = static_cast<double *>(info.ptr);

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
      auto *raw_tess = static_cast<egTessel *>(new_tess->blind);

      int ptype, pindex;
      double xyz[3];
      // std::cout << "old_raw_tess->nGlobal: " << old_raw_tess->nGlobal << "\n";
      // std::cout << "raw_tess->nGlobal: " << raw_tess->nGlobal << "\n";

      bool two_dimensional = false;
      if (new_coords_size == 2*old_raw_tess->nGlobal )
         two_dimensional = true;

      // std::cout << "new_coords_size: " << new_coords_size << "\n";
      // std::cout << "tess size: " << old_raw_tess->nGlobal << "\n";
      // std::cout << "tess size (2): " << 2 * (old_raw_tess->nGlobal / 3) << "\n";
      // std::cout << "two dim: " << two_dimensional << "\n";

      for (int i = 1; i <= old_raw_tess->nGlobal; ++i)
      {
         EG_getGlobal(new_tess, i, &ptype, &pindex, xyz);
         if (two_dimensional)
         {
            new_coords[(i-1)*2 + 0] = xyz[0];
            new_coords[(i-1)*2 + 1] = xyz[1];
         }
         else
         {
            new_coords[(i-1)*3 + 0] = xyz[0];
            new_coords[(i-1)*3 + 1] = xyz[1];
            new_coords[(i-1)*3 + 2] = xyz[2];
         }
         // std::cout << "(" << xyz[0] << ", " << xyz[1] << ", " << xyz[2] << ")\n";
      }




   }, "Map an existing surface tessalation to a new body with the same topology",
   py::arg("old_model"),
   py::arg("new_model"),
   py::arg("tess_file"),
   py::arg("new_coords"))
   ;

}

// #endif // MFEM_USE_PUMI
