#ifndef MACH_EGADS
#define MACH_EGADS

#include "mfem.hpp"
#include "adept.h"

#ifdef MFEM_USE_PUMI
#include "gmi_egads.h"

void getBoundaryNodeDisplacement(std::string something, apf::Mesh2* mesh, Array<mfem::Vector> disp_list);

#endif
#endif
