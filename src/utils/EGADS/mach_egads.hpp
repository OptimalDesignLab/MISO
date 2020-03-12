#ifndef MACH_EGADS
#define MACH_EGADS

/// try to put includes in the .cpp file unless theyre explicitly needed for
/// the function prototypes
#include "mfem.hpp"
#include "utils.hpp"
#include "egads.h"
#include "adept.h"

#ifdef MFEM_USE_PUMI
#include "gmi_egads.h"

void getBoundaryNodeDisplacement(std::string something, apf::Mesh2* mesh, mfem::Array<mfem::Vector> disp_list);

#endif
#endif
