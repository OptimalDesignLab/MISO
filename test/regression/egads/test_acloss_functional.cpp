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

// Commenting this test case out. Dead class. Not putting any more time into

///TODO: Determine if there is another way sigma should be modelled for this test
#include "../../../../test/unit/electromag_test_data.hpp" 
using namespace electromag_data;

using namespace std;
using namespace mfem;
using namespace mach;

namespace
{

// Commenting this test case out. Dead class. Not putting any more time into
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

// Previous test case (working for mfem::Coefficient sigma)
// TEST_CASE("HybridACLossFunctionalIntegrator::GetEnergy",
//           "[HybridACLossFunctionalIntegrator]")
// {

//    using namespace mfem;
//    // using namespace electromag_data;

//    const int dim = 3; // templating is hard here because mesh constructors
//    double delta = 1e-5;

//    if (!PCU_Comm_Initialized())
//       PCU_Comm_Init();
// #ifdef MFEM_USE_EGADS
//    gmi_register_egads();
//    gmi_egads_start();
// #endif
//    std::unique_ptr<apf::Mesh2, pumiDeleter> pumi_mesh(
//                            apf::loadMdsMesh("data/ac_cyl.egads", "data/ac_cyl.smb"));
   
//    pumi_mesh->verify();
//    apf::Numbering* aux_num = apf::createNumbering(pumi_mesh.get(), "aux_numbering",
//                                                   pumi_mesh->getShape(), 1);

//    apf::MeshIterator* it = pumi_mesh->begin(0);
//    apf::MeshEntity* v;
//    int count = 0;
//    while ((v = pumi_mesh->iterate(it)))
//    {
//      apf::number(aux_num, v, 0, 0, count++);
//    }
//    pumi_mesh->end(it);

//    ParPumiMesh mesh(MPI_COMM_WORLD, pumi_mesh.get());
//    mesh.EnsureNodes();

//    const auto p = 2;

//    std::unique_ptr<FiniteElementCollection> fec(
//       new ND_FECollection(p, dim));
//    std::unique_ptr<FiniteElementSpace> fes(new FiniteElementSpace(
//       &mesh, fec.get()));

//    NonlinearForm functional(fes.get());

//    // initialize state to give constant B field with mag 1
//    GridFunction A(fes.get());
//    VectorFunctionCoefficient field(3, [](const Vector& x, Vector &A)
//    {
//       A(0) = 0.5*x(1);
//       A(1) = -0.5*x(0);
//       A(2) = 0.0;
//    });
//    A.ProjectCoefficient(field);

//    // auto sigma_val = 58.14e6;
//    auto sigma_val = 1.0;
//    ConstantCoefficient sigma(sigma_val); // conductivity
//    auto frequency = 1000.0; // frequency
//    auto d = 0.348189415; // diameter of a strand
//    auto fill_factor = 0.6466; 
//    auto l = 2.50; // length
//    int n = 3;

//    functional.AddDomainIntegrator(
//       new mach::HybridACLossFunctionalIntegrator(sigma, frequency, d, fill_factor));

//    const auto b_mag = 1.0;
//    const auto loss = n * M_PI * l * std::pow(d, 4) * sigma_val
//                   * std::pow(2 * M_PI * frequency * b_mag, 2) / 128.0;
   
//    const double loss_fe = functional.GetEnergy(A);
//    std::cout << "functional loss: " << loss_fe << "\n";
//    std::cout << "analytical loss: " << loss << "\n";
//    const double loss_ratio = loss_fe / loss;
//    REQUIRE(loss_ratio == Approx(1.0).epsilon(1e-1));

// #ifdef MFEM_USE_PUMI
// #ifdef MFEM_USE_EGADS
//    gmi_egads_stop();
// #endif // MFEM_USE_EGADS
//    PCU_Comm_Free();
// #endif // MFEM_USE_PUMI

// }

// Commenting this test case out. Dead class. Not putting any more time into
// Now, adapting test case for StateCoefficient sigma
// TEST_CASE("HybridACLossFunctionalIntegrator::GetEnergy",
//           "[HybridACLossFunctionalIntegrator]")
// {

//    using namespace mfem;
//    // using namespace electromag_data;

//    const int dim = 3; // templating is hard here because mesh constructors
//    double delta = 1e-5;

//    if (!PCU_Comm_Initialized())
//       PCU_Comm_Init();
// #ifdef MFEM_USE_EGADS
//    gmi_register_egads();
//    gmi_egads_start();
// #endif
//    std::unique_ptr<apf::Mesh2, pumiDeleter> pumi_mesh(
//                            apf::loadMdsMesh("data/ac_cyl.egads", "data/ac_cyl.smb"));
   
//    pumi_mesh->verify();
//    apf::Numbering* aux_num = apf::createNumbering(pumi_mesh.get(), "aux_numbering",
//                                                   pumi_mesh->getShape(), 1);

//    apf::MeshIterator* it = pumi_mesh->begin(0);
//    apf::MeshEntity* v;
//    int count = 0;
//    while ((v = pumi_mesh->iterate(it)))
//    {
//      apf::number(aux_num, v, 0, 0, count++);
//    }
//    pumi_mesh->end(it);

//    ParPumiMesh mesh(MPI_COMM_WORLD, pumi_mesh.get());
//    mesh.EnsureNodes();

//    const auto p = 2;

//    std::unique_ptr<FiniteElementCollection> fec(
//       new ND_FECollection(p, dim));
//    std::unique_ptr<FiniteElementSpace> fes(new FiniteElementSpace(
//       &mesh, fec.get()));

//    NonlinearForm functional(fes.get());

//    // initialize state to give constant B field with mag 1
//    GridFunction A(fes.get());
//    VectorFunctionCoefficient field(3, [](const Vector& x, Vector &A)
//    {
//       A(0) = 0.5*x(1);
//       A(1) = -0.5*x(0);
//       A(2) = 0.0;
//    });
//    A.ProjectCoefficient(field);
//    std::cout << "L183\n";

//    // auto sigma_val = 58.14e6;
//    // auto sigma_val = 1.0;
//    auto sigma_val = 5.6497e7; // for 20 deg, using the new value from the material library
//    // auto sigma_val = 5.6497e7/(1+3.8e-3*(100-20)) ; // at 100 deg (when don't have a temperature field)
//    // Handling the coefficient for the sigma in the same way the StateCoefficient nu was handled in other tests
//    std::unique_ptr<mach::StateCoefficient> sigma(new SigmaCoefficient()); // parameters to make the sigma_val=1
//    std::cout << "L190\n";
//    // ConstantCoefficient sigma(sigma_val); // conductivity
//    auto frequency = 1000.0; // frequency
//    auto d = 0.348189415; // diameter of a strand
//    auto fill_factor = 0.6466; 
//    auto l = 2.50; // length
//    int n = 3;

//    //Function Coefficient model Representing the Temperature Field
//    GridFunction temp_field(fes.get());
//    VectorFunctionCoefficient Tfield_model(1, [](const Vector& x, Vector &T)
//    {
//       T(0) = 20;
//    });
//    // FunctionCoefficient Tfield_model(
//    //    [](const mfem::Vector &x)
//    //    {
//    //       std::cout << "x.Size()=" << x.Size() << "\n";
//    //       // x will be the point in space
//    //       double T = 0;
//    //       for (int i = 0; i < x.Size(); ++i)
//    //       {
//    //          T = 20; //constant temperature throughout mesh
//    //          // T = 77*x(0); // temperature linearly dependent in the x(0) direction
//    //          // T = 63*x(1); // temperature linearly dependent in the x(1) direction
//    //          // T = 30*std::pow(x(0),2); // temperature quadratically dependent in the x(0) direction
//    //          // T = 77*x(0)+63*x(1); // temperature linearly dependent in both x(0) and x(1) directions
//    //          // T = 30*std::pow(x(0),2) + 3*std::pow(x(1),2); // temperature quadratically dependent in both x(0) and x(1) directions

//    //       }
//    //       return T;
//    //       std::cout << "T=" << T << "\n";
//    //    });
//    temp_field.ProjectCoefficient(Tfield_model);
//    std::cout << "L218\n";

//    functional.AddDomainIntegrator(
//       new mach::HybridACLossFunctionalIntegrator(*sigma, frequency, d, fill_factor, &temp_field));
//    // functional.AddDomainIntegrator(
//    //    new mach::HybridACLossFunctionalIntegrator(*sigma, frequency, d, fill_factor)); // case where don't have a temperature field


//    std::cout << "L223\n";
//    const auto b_mag = 1.0;
//    const auto loss = n * M_PI * l * std::pow(d, 4) * sigma_val
//                   * std::pow(2 * M_PI * frequency * b_mag, 2) / 128.0;
   
//    const double loss_fe = functional.GetEnergy(A);
//    std::cout << "functional loss: " << loss_fe << "\n";
//    std::cout << "analytical loss: " << loss << "\n";
//    const double loss_ratio = loss_fe / loss;
//    REQUIRE(loss_ratio == Approx(1.0).epsilon(1e-1));

// #ifdef MFEM_USE_PUMI
// #ifdef MFEM_USE_EGADS
//    gmi_egads_stop();
// #endif // MFEM_USE_EGADS
//    PCU_Comm_Free();
// #endif // MFEM_USE_PUMI

// }
