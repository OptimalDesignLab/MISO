#include <fstream>
#include <iostream>

#include "mpi.h"
#include "catch.hpp"
#include "nlohmann/json.hpp"
#include "mfem.hpp"

#ifdef MFEM_USE_PUMI
#include "apfMDS.h"
#include "PCU.h"
#ifdef MFEM_USE_EGADS
#include "gmi_egads.h"
#endif // MFEM_USE_EGADS
#endif // MFEM_USE_PUMI

#include "electromag_integ.hpp"

using namespace std;
using namespace mfem;
using namespace mach;

namespace
{

/// specifies how to delete a PUMI mesh so that the PUMI mesh can be stored in
/// a unique_ptr and safely deleted
struct pumiDeleter
{
   void operator()(apf::Mesh2* mesh) const
   {
      mesh->destroyNative();
      apf::destroyMesh(mesh);
   }
};

}

TEST_CASE("HybridACLossFunctionalIntegrator::GetEnergy",
          "[HybridACLossFunctionalIntegrator]")
{

   using namespace mfem;
   // using namespace electromag_data;

   const int dim = 3; // templating is hard here because mesh constructors
   double delta = 1e-5;

   if (!PCU_Comm_Initialized())
      PCU_Comm_Init();
#ifdef MFEM_USE_EGADS
   gmi_register_egads();
   gmi_egads_start();
#endif
   std::unique_ptr<apf::Mesh2, pumiDeleter> pumi_mesh(
                           apf::loadMdsMesh("data/ac_cyl.egads", "data/ac_cyl.smb"));
   
   pumi_mesh->verify();
   apf::Numbering* aux_num = apf::createNumbering(pumi_mesh.get(), "aux_numbering",
                                                  pumi_mesh->getShape(), 1);

   apf::MeshIterator* it = pumi_mesh->begin(0);
   apf::MeshEntity* v;
   int count = 0;
   while ((v = pumi_mesh->iterate(it)))
   {
     apf::number(aux_num, v, 0, 0, count++);
   }
   pumi_mesh->end(it);

   ParPumiMesh mesh(MPI_COMM_WORLD, pumi_mesh.get());
   mesh.ReorientTetMesh();
   mesh.EnsureNodes();

   const auto p = 2;

   std::unique_ptr<FiniteElementCollection> fec(
      new ND_FECollection(p, dim));
   std::unique_ptr<FiniteElementSpace> fes(new FiniteElementSpace(
      &mesh, fec.get()));

   NonlinearForm functional(fes.get());

   // initialize state to give constant B field with mag 1
   GridFunction A(fes.get());
   VectorFunctionCoefficient field(3, [](const Vector& x, Vector &A)
   {
      A(0) = 0.5*x(1);
      A(1) = -0.5*x(0);
      A(2) = 0.0;
   });
   A.ProjectCoefficient(field);

   // auto sigma_val = 58.14e6;
   auto sigma_val = 1.0;
   ConstantCoefficient sigma(sigma_val); // conductivity
   auto frequency = 1000.0; // frequency
   auto d = 0.348189415; // diameter of a strand
   auto fill_factor = 0.6466; 
   auto l = 2.50; // length
   int n = 3;

   functional.AddDomainIntegrator(
      new mach::HybridACLossFunctionalIntegrator(sigma, frequency, d, fill_factor));

   const auto b_mag = 1.0;
   const auto loss = n * M_PI * l * std::pow(d, 4) * sigma_val
                  * std::pow(2 * M_PI * frequency * b_mag, 2) / 128.0;
   
   const double loss_fe = functional.GetEnergy(A);
   std::cout << "functional loss: " << loss_fe << "\n";
   std::cout << "analytical loss: " << loss << "\n";
   const double loss_ratio = loss_fe / loss;
   REQUIRE(loss_ratio == Approx(1.0).epsilon(1e-1));

#ifdef MFEM_USE_PUMI
#ifdef MFEM_USE_EGADS
   gmi_egads_stop();
#endif // MFEM_USE_EGADS
   PCU_Comm_Free();
#endif // MFEM_USE_PUMI

}
