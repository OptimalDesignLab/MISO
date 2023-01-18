#include "catch.hpp"
#include "mfem.hpp"

#include "electromag_test_data.hpp"

#include "coefficient.hpp"
///TODO: Once install mach again, replace the below line with simply: #include "electromag_integ.hpp"
#include "../../src/physics/electromagnetics/electromag_integ.hpp"
#include "material_library.hpp"

TEST_CASE("SteinmetzLossIntegrator::GetElementEnergy")
{
   const int dim = 3;
   int num_edge = 3;
   auto smesh = mfem::Mesh::MakeCartesian3D(num_edge, num_edge, num_edge,
                                            mfem::Element::TETRAHEDRON,
                                            1.0, 1.0, 1.0, true);

   mfem::ParMesh mesh(MPI_COMM_WORLD, smesh); 
   mesh.EnsureNodes();


   mfem::L2_FECollection fec(1, dim);
   mfem::ParFiniteElementSpace fes(&mesh, &fec);

   mfem::NonlinearForm functional(&fes);

   mfem::ConstantCoefficient rho(1.0);
   mfem::ConstantCoefficient k_s(0.01);
   mfem::ConstantCoefficient alpha(1.21);
   mfem::ConstantCoefficient beta(1.62);
   auto *integ = new mach::SteinmetzLossIntegrator(rho, k_s, alpha, beta);
   setInputs(*integ, {
      {"frequency", 151.0},
      {"max_flux_magnitude", 2.2}
   });

   functional.AddDomainIntegrator(integ);

   mfem::Vector dummy_vec(fes.GetTrueVSize());
   auto core_loss = functional.GetEnergy(dummy_vec);

   /// Answer should be k_s * pow(freq, alpha) * pow(|B|)^beta 
   /// 0.01 * pow(151, 1.21) * pow(2.2, 1.62) -> 15.5341269187
   REQUIRE(core_loss == Approx(15.5341269187));
}

TEST_CASE("SteinmetzLossIntegratorFreqSens::GetElementEnergy")
{
   using namespace mfem;
   using namespace electromag_data;

   double delta = 1e-5;

   // generate a 8 element mesh
   int num_edge = 2;
   auto mesh = Mesh::MakeCartesian2D(num_edge, num_edge,
                                     Element::TRIANGLE);
   mesh.EnsureNodes();
   const auto dim = mesh.SpaceDimension();

   auto f = [](const mfem::Vector &x)
   {
      double q = 0;
      for (int i = 0; i < x.Size(); ++i)
      {
         q += pow(x(i), 2);
      }
      return q;
   };

   auto f_rev_diff = [](const mfem::Vector &x, const double q_bar, mfem::Vector &x_bar)
   {
      for (int i = 0; i < x.Size(); ++i)
      {
         x_bar(i) += q_bar * 2 * x(i);
      }
   };

   mfem::FunctionCoefficient rho(f, f_rev_diff);
   mfem::FunctionCoefficient k_s(f, f_rev_diff);
   mfem::FunctionCoefficient alpha(f, f_rev_diff);
   mfem::FunctionCoefficient beta(f, f_rev_diff);
   // mfem::ConstantCoefficient rho(1.0);
   // mfem::ConstantCoefficient k_s(0.01);
   // mfem::ConstantCoefficient alpha(1.21);
   // mfem::ConstantCoefficient beta(1.62);

   for (int p = 1; p <= 4; ++p)
   {
      DYNAMIC_SECTION("...for degree p = " << p)
      {
         L2_FECollection fec(p, dim);
         FiniteElementSpace fes(&mesh, &fec);

         // initialize state
         GridFunction a(&fes);
         FunctionCoefficient pert(randState);
         a.ProjectCoefficient(pert);

         auto *integ = new mach::SteinmetzLossIntegrator(rho, k_s, alpha, beta);
         NonlinearForm functional(&fes);
         functional.AddDomainIntegrator(integ);

         // evaluate dJdp and compute its product with pert
         NonlinearForm dJdp(&fes);
         dJdp.AddDomainIntegrator(
            new mach::SteinmetzLossIntegratorFreqSens(*integ));

         double frequency = 2.0 + randNumber();
         mach::MachInputs inputs{
            {"frequency", frequency},
         };
         setInputs(*integ, inputs);
         double dfdp_fwd = dJdp.GetEnergy(a);

         // now compute the finite-difference approximation...
         inputs["frequency"] = frequency + delta;
         setInputs(*integ, inputs);
         double dfdp_fd_p = functional.GetEnergy(a);

         inputs["frequency"] = frequency - delta;
         setInputs(*integ, inputs);
         double dfdp_fd_m = functional.GetEnergy(a);

         double dfdp_fd = (dfdp_fd_p - dfdp_fd_m) / (2 * delta);

         REQUIRE(dfdp_fwd == Approx(dfdp_fd).margin(1e-8));

      }
   }
}

TEST_CASE("SteinmetzLossIntegratorMaxFluxSens::GetElementEnergy")
{
   using namespace mfem;
   using namespace electromag_data;

   double delta = 1e-5;

   // generate a 8 element mesh
   int num_edge = 2;
   auto mesh = Mesh::MakeCartesian2D(num_edge, num_edge,
                                     Element::TRIANGLE);
   mesh.EnsureNodes();
   const auto dim = mesh.SpaceDimension();

   auto f = [](const mfem::Vector &x)
   {
      double q = 0;
      for (int i = 0; i < x.Size(); ++i)
      {
         q += pow(x(i), 2);
      }
      return q;
   };

   auto f_rev_diff = [](const mfem::Vector &x, const double q_bar, mfem::Vector &x_bar)
   {
      for (int i = 0; i < x.Size(); ++i)
      {
         x_bar(i) += q_bar * 2 * x(i);
      }
   };

   mfem::FunctionCoefficient rho(f, f_rev_diff);
   mfem::FunctionCoefficient k_s(f, f_rev_diff);
   mfem::FunctionCoefficient alpha(f, f_rev_diff);
   mfem::FunctionCoefficient beta(f, f_rev_diff);
   // mfem::ConstantCoefficient rho(1.0);
   // mfem::ConstantCoefficient k_s(0.01);
   // mfem::ConstantCoefficient alpha(1.21);
   // mfem::ConstantCoefficient beta(1.62);

   for (int p = 1; p <= 4; ++p)
   {
      DYNAMIC_SECTION("...for degree p = " << p)
      {
         L2_FECollection fec(p, dim);
         FiniteElementSpace fes(&mesh, &fec);

         // initialize state
         GridFunction a(&fes);
         FunctionCoefficient pert(randState);
         a.ProjectCoefficient(pert);

         auto *integ = new mach::SteinmetzLossIntegrator(rho, k_s, alpha, beta);
         NonlinearForm functional(&fes);
         functional.AddDomainIntegrator(integ);

         // evaluate dJdp and compute its product with pert
         NonlinearForm dJdp(&fes);
         dJdp.AddDomainIntegrator(
            new mach::SteinmetzLossIntegratorMaxFluxSens(*integ));

         double max_flux_magnitude = 2.0 + randNumber();
         mach::MachInputs inputs{
            {"max_flux_magnitude", max_flux_magnitude},
         };
         setInputs(*integ, inputs);
         double dfdp_fwd = dJdp.GetEnergy(a);

         // now compute the finite-difference approximation...
         inputs["max_flux_magnitude"] = max_flux_magnitude + delta;
         setInputs(*integ, inputs);
         double dfdp_fd_p = functional.GetEnergy(a);

         inputs["max_flux_magnitude"] = max_flux_magnitude - delta;
         setInputs(*integ, inputs);
         double dfdp_fd_m = functional.GetEnergy(a);

         double dfdp_fd = (dfdp_fd_p - dfdp_fd_m) / (2 * delta);

         REQUIRE(dfdp_fwd == Approx(dfdp_fd).margin(1e-8));

      }
   }
}

TEST_CASE("SteinmetzLossIntegratorMeshSens::AssembleRHSElementVect")
{
   using namespace mfem;
   using namespace electromag_data;

   double delta = 1e-5;

   // generate a 8 element mesh
   int num_edge = 2;
   auto mesh = Mesh::MakeCartesian2D(num_edge, num_edge,
                                     Element::TRIANGLE);
   mesh.EnsureNodes();
   const auto dim = mesh.SpaceDimension();

   auto f = [](const mfem::Vector &x)
   {
      double q = 0;
      for (int i = 0; i < x.Size(); ++i)
      {
         q += pow(x(i), 2);
      }
      return q;
   };

   auto f_rev_diff = [](const mfem::Vector &x, const double q_bar, mfem::Vector &x_bar)
   {
      for (int i = 0; i < x.Size(); ++i)
      {
         x_bar(i) += q_bar * 2 * x(i);
      }
   };

   mfem::FunctionCoefficient rho(f, f_rev_diff);
   mfem::FunctionCoefficient k_s(f, f_rev_diff);
   mfem::FunctionCoefficient alpha(f, f_rev_diff);
   mfem::FunctionCoefficient beta(f, f_rev_diff);

   for (int p = 1; p <= 4; ++p)
   {
      DYNAMIC_SECTION("...for degree p = " << p)
      {
         L2_FECollection fec(p, dim);
         FiniteElementSpace fes(&mesh, &fec);

         // initialize state
         GridFunction a(&fes);
         FunctionCoefficient pert(randState);
         a.ProjectCoefficient(pert);

         auto *integ = new mach::SteinmetzLossIntegrator(rho, k_s, alpha, beta);
         NonlinearForm functional(&fes);
         functional.AddDomainIntegrator(integ);

         // extract mesh nodes and get their finite-element space
         auto &x_nodes = *mesh.GetNodes();
         auto &mesh_fes = *x_nodes.FESpace();

         // create v displacement field
         GridFunction v(&mesh_fes);
         VectorFunctionCoefficient v_pert(dim, randVectorState);
         v.ProjectCoefficient(v_pert);

         // initialize the vector that dJdx multiplies
         GridFunction p(&mesh_fes);
         p.ProjectCoefficient(v_pert);

         // evaluate dJdx and compute its product with p
         LinearForm dJdx(&mesh_fes);
         dJdx.AddDomainIntegrator(
            new mach::SteinmetzLossIntegratorMeshSens(a, *integ));
         dJdx.Assemble();
         double dJdx_dot_p = dJdx * p;

         // now compute the finite-difference approximation...
         GridFunction x_pert(x_nodes);
         x_pert.Add(-delta, p);
         mesh.SetNodes(x_pert);
         fes.Update();
         double dJdx_dot_p_fd = -functional.GetEnergy(a);
         x_pert.Add(2 * delta, p);
         mesh.SetNodes(x_pert);
         fes.Update();
         dJdx_dot_p_fd += functional.GetEnergy(a);
         dJdx_dot_p_fd /= (2 * delta);
         mesh.SetNodes(x_nodes); // remember to reset the mesh nodes
         fes.Update();

         REQUIRE(dJdx_dot_p == Approx(dJdx_dot_p_fd));
      }
   }
}

// Added test case for SteinmetzLossDistributionIntegrator
///TODO: Finish test case in conjunction with implementation itself
TEST_CASE("SteinmetzLossDistributionIntegrator::AssembleRHSElementVect")
{
   using namespace mfem;
   using namespace electromag_data;

   double delta = 1e-5;

   // generate a 8 element mesh, simple 2D domain, 0<=x<=1, 0<=y<=1
   int num_edge = 2;
   auto mesh = Mesh::MakeCartesian2D(num_edge, num_edge,
                                     Element::TRIANGLE);
   mesh.EnsureNodes();
   const auto dim = mesh.SpaceDimension();
   
   // Set the Steinmetz Coefficients
   mfem::ConstantCoefficient rho(1.0);
   mfem::ConstantCoefficient k_s(0.01);
   mfem::ConstantCoefficient alpha(1.21);
   mfem::ConstantCoefficient beta(1.62);
   
   // Adapted from TEST_CASE("DCLossFunctionalDistributionIntegrator::AssembleRHSElementVect (2D)"):
   for (int p = 1; p <= 4; ++p)
   {
      DYNAMIC_SECTION("...for degree p = " << p)
      {
         // mesh.SetCurvature(p);

         H1_FECollection fec(p, dim);
         FiniteElementSpace fes(&mesh, &fec);

         // initialize state
         GridFunction a(&fes);
         FunctionCoefficient pert(randState);
         a.ProjectCoefficient(pert);

         // extract mesh nodes and get their finite-element space
         auto &x_nodes = *mesh.GetNodes();
         auto &mesh_fes = *x_nodes.FESpace();

         // create v displacement field
         GridFunction v(&mesh_fes);
         VectorFunctionCoefficient v_pert(dim, randVectorState);
         v.ProjectCoefficient(v_pert);

         // initialize the vector that dJdx multiplies
         GridFunction p(&mesh_fes);
         p.ProjectCoefficient(v_pert);
         
         // From SteinmetzLossIntegrator::GetElementEnergy
         // auto *integ = new mach::SteinmetzLossIntegrator;
         // setInputs(*integ, {
         //    {"frequency", 151.0},
         //    {"max_flux_magnitude", 2.2}
         // });

         // evaluate dJdx and compute its product with p
         LinearForm dJdx(&mesh_fes);
         dJdx.AddDomainIntegrator(
            new mach::SteinmetzLossDistributionIntegrator(rho, k_s, alpha, beta));         
         dJdx.Assemble();
         std::cout << "dJdx.Size() = " << dJdx.Size() << "p.Size() = " << p.Size() << "\n";
         double dJdx_dot_p = dJdx * p;
         std::cout << "dJdx_dot_p=" << dJdx_dot_p << "\n";

         ///TODO: compute the finite-difference approximation or other value as needed to assert LinearForm
                 
         ///TODO: Add Assertion
         // std::cout << "dJdx_dot_p_fd=" << dJdx_dot_p_fd << "\n";
         // REQUIRE(dJdx_dot_p == Approx(dJdx_dot_p_fd));
      }
   }   
}

// Adding CAL2 Core Loss Integrator test here (can always make a new/separate test file)
// Revise test to allow for the maximum flux value to be passed in (value only used if no peak flux field) 
TEST_CASE("CAL2CoreLossIntegrator::GetElementEnergy")
{
   using namespace mfem;
   using namespace electromag_data;

   
   const int dim = 2;
   // if (dim==2)
   // {
   // Option 1: Generate an 8 element mesh in 2D

   int num_edge = 2;
   auto mesh = Mesh::MakeCartesian2D(num_edge, num_edge,
                                    Element::TRIANGLE);
   mesh.EnsureNodes();

   
   // }
   // else if (dim==3)
   // {
   // // Option 2: Generate a 3D unit cube mesh
   
   // int num_edge = 3;
   // auto smesh = Mesh::MakeCartesian3D(num_edge, num_edge, num_edge,
   //                                        Element::TETRAHEDRON,
   //                                        1.0, 1.0, 1.0, true);

   // ParMesh mesh(MPI_COMM_WORLD, smesh); 
   // mesh.EnsureNodes();

   // L2_FECollection fec(p, dim); // Stick with L2 elements or use other FEs? 
   // ParFiniteElementSpace fes(&mesh, &fec);
   // }

   //Function Coefficient model Representing the B Field (peak flux density in this case)
   mfem::FunctionCoefficient Bfield_model(
      [](const mfem::Vector &x)
      {
         // x will be the point in space
         double B = 0;
         for (int i = 0; i < x.Size(); ++i)
         {
            B = 2.2; //constant flux density throughout mesh
            // B = 2.4*x(0); // flux density linearly dependent in the x(0) direction
            // B = 1.1*x(1); // flux density linearly dependent in the x(1) direction
            // B = 3.0*std::pow(x(0),2); // flux density quadratically dependent in the x(0) direction
            // B = 2.4*x(0)+1.1*x(1); // flux density linearly dependent in both x(0) and x(1) directions
            // B = 3.0*std::pow(x(0),2) + 0.3*std::pow(x(1),2); // flux density quadratically dependent in both x(0) and x(1) directions

         }
         return B;
      });

   //Function Coefficient model Representing the Temperature Field
   mfem::FunctionCoefficient Tfield_model(
      [](const mfem::Vector &x)
      {
         // x will be the point in space
         double T = 0;
         for (int i = 0; i < x.Size(); ++i)
         {
            T = 37; //constant temperature throughout mesh
            // T = 77*x(0); // temperature linearly dependent in the x(0) direction
            // T = 63*x(1); // temperature linearly dependent in the x(1) direction
            // T = 30*std::pow(x(0),2); // temperature quadratically dependent in the x(0) direction
            // T = 77*x(0)+63*x(1); // temperature linearly dependent in both x(0) and x(1) directions
            // T = 30*std::pow(x(0),2) + 3*std::pow(x(1),2); // temperature quadratically dependent in both x(0) and x(1) directions

         }
         return T;
      });

   // Loop over various degrees of elements (1 to 4)
   for (int p = 1; p <= 4; ++p)
   {
      DYNAMIC_SECTION("...for degree p = " << p)
      {
   
      // Create the finite element collection and finite element space for the current order
      H1_FECollection fec(p, dim);
      FiniteElementSpace fes(&mesh, &fec);

      // extract mesh nodes and get their finite-element space
      auto &x_nodes = *mesh.GetNodes();
      auto &mesh_fes = *x_nodes.FESpace();

      mfem::NonlinearForm functional(&fes);

      // Set the density
      mfem::ConstantCoefficient rho(1.0);

      // ** Made CAL2Coefficient simpler to avoid the parameters. Define CAL2_kh (similar methodology to SigmaCoefficient from test_electromag_integ)
      // double T0 = 20;
      // double T1 = 200;
      // std::vector<double> kh_T0 = {0.0997091541786544,
      //          -0.129193571991623,
      //          0.0900090637806644,
      //          -0.0212834836667556};
      // std::vector<double> kh_T1 = {0.0895406177349016,
      //          -0.0810594723247055,
      //          0.0377588555136910,
      //          -0.00511339186996760};
      std::unique_ptr<mach::ThreeStateCoefficient> CAL2_kh(new CAL2Coefficient());

      // ** Made CAL2Coefficient simpler to avoid the parameters. Define CAL2_ke (similar methodology to SigmaCoefficient from test_electromag_integ)
      // T0=20 and T1=200 once again
      // std::vector<double> ke_T0 = {-5.93693970727006e-06,
      //          0.000117138629373709,
      //          -0.000130355460369590,
      //          4.10973552619398e-05};
      // std::vector<double> ke_T1 = {1.79301614571386e-05,
      //          7.45159671115992e-07,
      //          -1.19410662547280e-06,
      //          3.53133402660246e-07};
      std::unique_ptr<mach::ThreeStateCoefficient> CAL2_ke(new CAL2Coefficient());

      // Create the temperature_field grid function by mapping the function coefficient to a grid function
      mfem::GridFunction temperature_field(&fes);
      temperature_field.ProjectCoefficient(Tfield_model);

      // Create the peak flux field (B) grid function by mapping the function coefficient to a grid function
      mfem::GridFunction peak_flux(&fes);
      peak_flux.ProjectCoefficient(Bfield_model);
      
      double f = 1000.0; double Bm = 1.7;
      auto *integ = new mach::CAL2CoreLossIntegrator(rho, *CAL2_kh, *CAL2_ke, peak_flux, &temperature_field);
      setInputs(*integ, {
         {"frequency", f},
         {"max_flux_magnitude", Bm}
      });
      // auto *integ = new mach::CAL2CoreLossIntegrator(rho, *CAL2_kh, *CAL2_ke, peak_flux);
      // setInputs(*integ, {
      //    {"frequency", f},
      //    {"max_flux_magnitude", Bm}
      // }); // for the case where the temperature_field is a null pointer (not passed in)

      functional.AddDomainIntegrator(integ);

      mfem::Vector dummy_vec(fes.GetTrueVSize());
      auto CAL2_core_loss = functional.GetEnergy(dummy_vec);
      // std::cout << "CAL2_core_loss=" << CAL2_core_loss << "\n";

      double Expected_core_loss = f*37*Bm + pow(f,2)*37*Bm; // for B=1.7, T=37 (both const), UseMaxFluxValueAndNotPeakFluxField = true (passes for all degrees)
      // double Expected_core_loss = f*100*Bm + pow(f,2)*100*Bm; // for B=1.7, T=100 (no temp field), UseMaxFluxValueAndNotPeakFluxField = true (passes for all degrees)
      // double Expected_core_loss = f*(77.0/2)*Bm + pow(f,2)*(77.0/2)*Bm; // for B=1.7, T=77*x(0), UseMaxFluxValueAndNotPeakFluxField = true
      // double Expected_core_loss = f*(30.0/3)*Bm + pow(f,2)*(30.0/3)*Bm; // for B=1.7, T=30*std::pow(x(0),2), UseMaxFluxValueAndNotPeakFluxField = true
      // double Expected_core_loss = f*37*2.2 + pow(f,2)*37*2.2; // for B=2.2, T=37, UseMaxFluxValueAndNotPeakFluxField = false
      // double Expected_core_loss = f*37*(2.4/2) + pow(f,2)*37*(2.4/2); // for B=2.4*x(0), T=37, UseMaxFluxValueAndNotPeakFluxField = false
      // double Expected_core_loss = f*37*(3.0/3) + pow(f,2)*37*(3.0/3); // for B=3.0*std::pow(x(0),2), T=37, UseMaxFluxValueAndNotPeakFluxField = false
      
      // std::cout << "core_loss_diff = " << CAL2_core_loss-Expected_core_loss << "\n";
      REQUIRE(CAL2_core_loss == Approx(Expected_core_loss));
      // The below was for when CAL2Coefficient was not as simple (previously) 
      // At B=1.7, T=20 (both const): CAL2_kh=0.03564052086424506, CAL2_ke=1.8382756161830418e-05, pFe=(0.03564052086424506)*1000*pow(1.7,2)+(1.8382756161830418e-05)*pow(1000,2)*pow(1.7,2)=156.12727060535812 W
      // Expected_core_loss = 156.12727060535812; // passes for all orders of p, including p=1
      // At B=1.7, T=100 (both const): CAL2_kh=0.03568496179583322, CAL2_ke=1.798193527110098e-05, pFe=(0.03568496179583322)*1000*pow(1.7,2)+(1.798193527110098e-05)*pow(1000,2)*pow(1.7,2)=155.0973325234398 W
      // Expected_core_loss = 155.0973325234398; // passes for all orders of p, including p=1
      // At B=2.2, T=37 (both const): CAL2_kh=0.025918677125662013, CAL2_ke=5.4589283749123626e-05, pFe=(0.025918677125662013)*1000*pow(2.2,2)+(5.4589283749123626e-05)*pow(1000,2)*pow(2.2,2)=389.6585306339625 W
      // Expected_core_loss = 389.6585306339625; // passes for all orders of p, including p=1
      // At B=2.2, T=77*x(0): CAL2_kh=0.026043798894524624, CAL2_ke=5.424843329564548e-05, pFe=(0.026043798894524624)*1000*pow(2.2,2)+(5.424843329564548e-05)*pow(1000,2)*pow(2.2,2)=388.61440380042336 W
      // Expected_core_loss = 388.61440380042336; // passes for all orders of p, including p=1
      // At B=2.2, T=63*x(1): CAL2_kh=0.02545989730649911, CAL2_ke=5.583906874521016e-05, pFe=(0.02545989730649911)*1000*pow(2.2,2)+(5.583906874521016e-05)*pow(1000,2)*pow(2.2,2)=393.4869956902729 W
      // Expected_core_loss = 393.4869956902729; // passes for all orders of p, including p=1
      // Temporarily adjusting logic in CAL2CLI to have CAL2_kh and CAL2_ke=1 (temporarily)
      // With B=2.4*x(0), T=37: CAL2_kh and CAL2_ke=1 (temporarily), pFe=1.92192e6 W (analytical calc, WolframAlpha verified)
      // Expected_core_loss = 1.92192e6; // as expected, fails for p=1 and passes for p=2
      }
   }
}

// Added test case for CAL2CoreLossIntegratorFreqSens::GetElementEnergy
TEST_CASE("CAL2CoreLossIntegratorFreqSens::GetElementEnergy")
{
   using namespace mfem;
   using namespace electromag_data;

   double delta = 1e-5;

   // generate a 8 element mesh
   int num_edge = 2;
   auto mesh = Mesh::MakeCartesian2D(num_edge, num_edge,
                                     Element::TRIANGLE);
   mesh.EnsureNodes();
   const auto dim = mesh.SpaceDimension();

   auto f = [](const mfem::Vector &x)
   {
      double q = 0;
      for (int i = 0; i < x.Size(); ++i)
      {
         q += pow(x(i), 2);
      }
      return q;
   };

   auto f_rev_diff = [](const mfem::Vector &x, const double q_bar, mfem::Vector &x_bar)
   {
      for (int i = 0; i < x.Size(); ++i)
      {
         x_bar(i) += q_bar * 2 * x(i);
      }
   };

   // Set the density
   mfem::FunctionCoefficient rho(f, f_rev_diff);

   for (int p = 1; p <= 4; ++p)
   {
      DYNAMIC_SECTION("...for degree p = " << p)
      {
         L2_FECollection fec(p, dim);
         FiniteElementSpace fes(&mesh, &fec);

         // initialize state (used for both peak flux (if using field) and temperature fields)
         GridFunction a(&fes);
         FunctionCoefficient pert(randState);
         a.ProjectCoefficient(pert);

         // Set the CAL2 coefficients
         std::unique_ptr<mach::ThreeStateCoefficient> CAL2_kh(new CAL2Coefficient());
         std::unique_ptr<mach::ThreeStateCoefficient> CAL2_ke(new CAL2Coefficient());

         auto *integ = new mach::CAL2CoreLossIntegrator(rho, *CAL2_kh, *CAL2_ke, a, &a);
         // auto *integ = new mach::CAL2CoreLossIntegrator(rho, *CAL2_kh, *CAL2_ke, a); // for the case where the temperature_field is a null pointer (not passed in)
         NonlinearForm functional(&fes);
         functional.AddDomainIntegrator(integ);

         // evaluate dJdp and compute its product with pert
         NonlinearForm dJdp(&fes);
         dJdp.AddDomainIntegrator(
            new mach::CAL2CoreLossIntegratorFreqSens(*integ));

         double frequency = 2.0 + randNumber();
         double Bm = 1.7;
         mach::MachInputs inputs{
            {"frequency", frequency},
            {"max_flux_magnitude", Bm}
         };
         setInputs(*integ, inputs);
         double dfdp_fwd = dJdp.GetEnergy(a);
         // std::cout << "dfdp_fwd = " << dfdp_fwd << "\n";

         // now compute the finite-difference approximation...
         inputs["frequency"] = frequency + delta;
         setInputs(*integ, inputs);
         double dfdp_fd_p = functional.GetEnergy(a);

         inputs["frequency"] = frequency - delta;
         setInputs(*integ, inputs);
         double dfdp_fd_m = functional.GetEnergy(a);

         double dfdp_fd = (dfdp_fd_p - dfdp_fd_m) / (2 * delta);

         // std::cout << "dfdp_fwd = " << dfdp_fwd << "\n";
         REQUIRE(dfdp_fwd == Approx(dfdp_fd).margin(1e-8));
      }
   }
}

// Added test case for CAL2CoreLossIntegratorMaxFluxSens:: GetElementEnergy
TEST_CASE("CAL2CoreLossIntegratorMaxFluxSens::GetElementEnergy")
{
   using namespace mfem;
   using namespace electromag_data;

   double delta = 1e-5;

   // generate a 8 element mesh
   int num_edge = 2;
   auto mesh = Mesh::MakeCartesian2D(num_edge, num_edge,
                                     Element::TRIANGLE);
   mesh.EnsureNodes();
   const auto dim = mesh.SpaceDimension();

   auto f = [](const mfem::Vector &x)
   {
      double q = 0;
      for (int i = 0; i < x.Size(); ++i)
      {
         q += pow(x(i), 2);
      }
      return q;
   };

   auto f_rev_diff = [](const mfem::Vector &x, const double q_bar, mfem::Vector &x_bar)
   {
      for (int i = 0; i < x.Size(); ++i)
      {
         x_bar(i) += q_bar * 2 * x(i);
      }
   };

   // Set the density
   mfem::FunctionCoefficient rho(f, f_rev_diff);

   for (int p = 1; p <= 4; ++p)
   {
      DYNAMIC_SECTION("...for degree p = " << p)
      {
         L2_FECollection fec(p, dim);
         FiniteElementSpace fes(&mesh, &fec);

         // initialize state (used for both peak flux (if using field) and temperature fields)
         GridFunction a(&fes);
         FunctionCoefficient pert(randState);
         a.ProjectCoefficient(pert);

         // Set the CAL2 coefficients
         std::unique_ptr<mach::ThreeStateCoefficient> CAL2_kh(new CAL2Coefficient());
         std::unique_ptr<mach::ThreeStateCoefficient> CAL2_ke(new CAL2Coefficient());

         auto *integ = new mach::CAL2CoreLossIntegrator(rho, *CAL2_kh, *CAL2_ke, a, &a);
         // auto *integ = new mach::CAL2CoreLossIntegrator(rho, *CAL2_kh, *CAL2_ke, a); // for the case where the temperature_field is a null pointer (not passed in)
         NonlinearForm functional(&fes);
         functional.AddDomainIntegrator(integ);

         // evaluate dJdp and compute its product with pert
         NonlinearForm dJdp(&fes);
         dJdp.AddDomainIntegrator(
            new mach::CAL2CoreLossIntegratorMaxFluxSens(*integ));

         double frequency = 2.0;
         double Bm = 1.7 + randNumber();
         mach::MachInputs inputs{
            {"frequency", frequency},
            {"max_flux_magnitude", Bm}
         };
         setInputs(*integ, inputs);
         double dfdp_fwd = dJdp.GetEnergy(a);
         std::cout << "dfdp_fwd = " << dfdp_fwd << "\n";

         // now compute the finite-difference approximation...
         inputs["max_flux_magnitude"] = Bm + delta;
         setInputs(*integ, inputs);
         double dfdp_fd_p = functional.GetEnergy(a);

         inputs["max_flux_magnitude"] = Bm - delta;
         setInputs(*integ, inputs);
         double dfdp_fd_m = functional.GetEnergy(a);

         double dfdp_fd = (dfdp_fd_p - dfdp_fd_m) / (2 * delta);

         std::cout << "dfdp_fwd = " << dfdp_fwd << "\n";
         REQUIRE(dfdp_fwd == Approx(dfdp_fd).margin(1e-8));
      }
   }
}

// Added test case for CAL2CoreLossIntegratorTemperatureSens::GetElementEnergy
TEST_CASE("CAL2CoreLossIntegratorTemperatureSens::GetElementEnergy")
{
   using namespace mfem;
   using namespace electromag_data;

   double delta = 1e-5;

   // generate a 8 element mesh
   int num_edge = 2;
   auto mesh = Mesh::MakeCartesian2D(num_edge, num_edge,
                                     Element::TRIANGLE);
   mesh.EnsureNodes();
   const auto dim = mesh.SpaceDimension();

   auto f = [](const mfem::Vector &x)
   {
      double q = 0;
      for (int i = 0; i < x.Size(); ++i)
      {
         std::cout << "x(" << i << ")=" << x(i) <<", ";
         q += pow(x(i), 2);
      }
      std::cout << "q=" << q << "\n";
      return q;
   };

   auto f_rev_diff = [](const mfem::Vector &x, const double q_bar, mfem::Vector &x_bar)
   {
      std::cout << "f_rev_diff being called for temp sens\n";
      for (int i = 0; i < x.Size(); ++i)
      {
         x_bar(i) += q_bar * 2 * x(i);
      }
   };

   // Set the density
   mfem::FunctionCoefficient rho(f, f_rev_diff);

   ///TODO: Revert to (int p = 1; p <= 4; ++p)
   for (int p = 1; p <= 1; ++p)
   {
      DYNAMIC_SECTION("...for degree p = " << p)
      {
         // L2_FECollection fec(p, dim);
         H1_FECollection fec(p, dim);
         FiniteElementSpace fes(&mesh, &fec);

         // extract mesh nodes and get their finite-element space
         auto &x_nodes = *mesh.GetNodes();
         auto &mesh_fes = *x_nodes.FESpace();

         // initialize state (used for peak flux (if using field))
         GridFunction a(&fes);
         FunctionCoefficient pert(randState);
         a.ProjectCoefficient(pert);

         // Set the temperature field
         GridFunction temperature_field(&fes);
         FunctionCoefficient temperature_func(f, f_rev_diff);
         temperature_field.ProjectCoefficient(temperature_func);

         ///TODO: Remove comment out once done debugging
         // for ( int i = 0; i < mesh.GetNV(); i++ )
         // {
         //    for ( int j = 0; j < 2; j++ )
         //    {
         //          std::cout << x_nodes.Elem(i + j*mesh.GetNV()) << " ";
         //    }
         //    std::cout << "\n";
         // }
         // for (int j = 0; j < temperature_field.Size(); j++)
         // {
         //    std::cout << "At node " << x_nodes.Elem(j) << "temperature=" << temperature_field.Elem(j) << "\n";
         // }
          
         
         // Set the CAL2 coefficients
         std::unique_ptr<mach::ThreeStateCoefficient> CAL2_kh(new CAL2Coefficient());
         std::unique_ptr<mach::ThreeStateCoefficient> CAL2_ke(new CAL2Coefficient());

         auto *integ = new mach::CAL2CoreLossIntegrator(rho, *CAL2_kh, *CAL2_ke, a, &temperature_field);
         // // auto *integ = new mach::CAL2CoreLossIntegrator(rho, *CAL2_kh, *CAL2_ke, a); // for the case where the temperature_field is a null pointer (not passed in)
         NonlinearForm functional(&fes);
         functional.AddDomainIntegrator(integ);

         // evaluate dJdp and compute its product with pert
         NonlinearForm dJdp(&fes);
         dJdp.AddDomainIntegrator(
            new mach::CAL2CoreLossIntegratorTemperatureSens(*integ));

         double frequency = 2.0;
         double Bm = 1.7;
         mach::MachInputs inputs{
            {"frequency", frequency},
            {"max_flux_magnitude", Bm}
         };
         setInputs(*integ, inputs);
         double dfdp_fwd = dJdp.GetEnergy(a);
         std::cout << "dfdp_fwd = " << dfdp_fwd << "\n";

         ///TODO: Add appropriate assertion
         // // now compute the finite-difference approximation...
         // inputs["frequency"] = frequency + delta;
         // setInputs(*integ, inputs);
         // double dfdp_fd_p = functional.GetEnergy(a);

         // inputs["frequency"] = frequency - delta;
         // setInputs(*integ, inputs);
         // double dfdp_fd_m = functional.GetEnergy(a);

         // double dfdp_fd = (dfdp_fd_p - dfdp_fd_m) / (2 * delta);

         // // std::cout << "dfdp_fwd = " << dfdp_fwd << "\n";
         // REQUIRE(dfdp_fwd == Approx(dfdp_fd).margin(1e-8));
      }
   }
}

// Added test case for CAL2CoreLossDistributionIntegrator
///TODO: Finish test case in conjunction with implementation itself
TEST_CASE("CAL2CoreLossDistributionIntegrator::AssembleRHSElementVect")
{
   using namespace mfem;
   using namespace electromag_data;

   double delta = 1e-5;

   // generate a 8 element mesh, simple 2D domain, 0<=x<=1, 0<=y<=1
   int num_edge = 2;
   auto mesh = Mesh::MakeCartesian2D(num_edge, num_edge,
                                     Element::TRIANGLE);
   mesh.EnsureNodes();
   const auto dim = mesh.SpaceDimension();
   
   // Set the density
   mfem::ConstantCoefficient rho(1.0);

   // Adapted from TEST_CASE("DCLossFunctionalDistributionIntegrator::AssembleRHSElementVect (2D)"):
   for (int p = 1; p <= 4; ++p)
   {
      DYNAMIC_SECTION("...for degree p = " << p)
      {
         // mesh.SetCurvature(p);

         H1_FECollection fec(p, dim);
         FiniteElementSpace fes(&mesh, &fec);

         // initialize state (used for both peak flux (if using field) and temperature fields)
         GridFunction a(&fes);
         FunctionCoefficient pert(randState);
         a.ProjectCoefficient(pert);

         // extract mesh nodes and get their finite-element space
         auto &x_nodes = *mesh.GetNodes();
         auto &mesh_fes = *x_nodes.FESpace();

         // create v displacement field
         GridFunction v(&mesh_fes);
         VectorFunctionCoefficient v_pert(dim, randVectorState);
         v.ProjectCoefficient(v_pert);

         // initialize the vector that dJdx multiplies
         GridFunction p(&mesh_fes);
         p.ProjectCoefficient(v_pert);

         // Set up the CAL2 Coefficients
         std::unique_ptr<mach::ThreeStateCoefficient> CAL2_kh(new CAL2Coefficient());
         std::unique_ptr<mach::ThreeStateCoefficient> CAL2_ke(new CAL2Coefficient());

         // Set up the integrator
         auto *integ = new mach::CAL2CoreLossDistributionIntegrator(rho, *CAL2_kh, *CAL2_ke, a, &a);
         // auto *integ = new mach::CAL2CoreLossDistributionIntegrator(rho, *CAL2_kh, *CAL2_ke, a); // for the case where the temperature_field is a null pointer (not passed in)
         setInputs(*integ, {
            {"frequency", 1000.0},
            {"max_flux_magnitude", 2.0}
         });

         // evaluate dJdx and compute its product with p
         LinearForm dJdx(&mesh_fes);
         dJdx.AddDomainIntegrator(integ);         
         dJdx.Assemble();
         double dJdx_dot_p = dJdx * p;
         std::cout << "CAL2CLDI dJdx_dot_p=" << dJdx_dot_p << "\n";

         ///TODO: compute the finite-difference approximation or other value as needed to assert LinearForm
                 
         ///TODO: Add Assertion
         // std::cout << "dJdx_dot_p_fd=" << dJdx_dot_p_fd << "\n";
         // REQUIRE(dJdx_dot_p == Approx(dJdx_dot_p_fd));
      }
   }   
}

/** not maintaining anymore
TEST_CASE("DomainResIntegrator::AssembleElementVector",
          "[DomainResIntegrator for Steinmetz]")
{
   using namespace mfem;
   using namespace euler_data;
   using namespace mach;

   const int dim = 3; // templating is hard here because mesh constructors
   double delta = 1e-5;

   // generate a 8 element mesh
   int num_edge = 1;
   std::unique_ptr<Mesh> mesh = electromag_data::getMesh(2, 1);
                              //(new Mesh(num_edge, num_edge, num_edge, Element::TETRAHEDRON,
                              //        true , 1.0, 1.0, 1.0, true));
   mesh->EnsureNodes();

   for (int p = 1; p <= 4; ++p)
   {
      DYNAMIC_SECTION("...for degree p = " << p)
      {
         // get the finite-element space for the state and adjoint
         H1_FECollection fec(p, dim);
         ND_FECollection feca(p, dim);
         FiniteElementSpace fes(mesh.get(), &fec);
         FiniteElementSpace fesa(mesh.get(), &feca);

         // we use res for finite-difference approximation
         GridFunction A(&fesa);
         VectorFunctionCoefficient perta(dim, electromag_data::randVectorState);
         A.ProjectCoefficient(perta);
         // std::unique_ptr<Coefficient> q2(new FunctionCoefficient(func, funcRevDiff));
         std::unique_ptr<Coefficient> q2(new SteinmetzCoefficient(
                        1, 2, 4, 0.5, 0.6, A));
         // std::unique_ptr<mach::MeshDependentCoefficient> Q;
         // Q.reset(new mach::MeshDependentCoefficient());
         // Q->addCoefficient(1, move(q1)); 
         // Q->addCoefficient(2, move(q2));
         LinearForm res(&fes);
         res.AddDomainIntegrator(
            new DomainLFIntegrator(*q2));

         // initialize state and adjoint; here we randomly perturb a constant state
         GridFunction adjoint(&fes);
         FunctionCoefficient pert(electromag_data::randState);
         adjoint.ProjectCoefficient(pert);

         // extract mesh nodes and get their finite-element space
         auto *x_nodes = dynamic_cast<GridFunction*>(mesh->GetNodes());
         auto *mesh_fes = dynamic_cast<FiniteElementSpace*>(x_nodes->FESpace());

         // build the nonlinear form for d(psi^T R)/dx 
         NonlinearForm dfdx_form(mesh_fes);
         dfdx_form.AddDomainIntegrator(
            new mach::DomainResIntegrator(*q2, &adjoint));

         // initialize the vector that we use to perturb the mesh nodes
         GridFunction v(mesh_fes);
         VectorFunctionCoefficient v_rand(dim, electromag_data::randVectorState);
         v.ProjectCoefficient(v_rand);

         // evaluate df/dx and contract with v
         GridFunction dfdx(*x_nodes);
         dfdx_form.Mult(*x_nodes, dfdx);
         double dfdx_v = dfdx * v;

         // now compute the finite-difference approximation...
         GridFunction x_pert(*x_nodes);
         GridFunction r(&fes);
         x_pert.Add(delta, v);
         mesh->SetNodes(x_pert);
         res.Assemble();
         double dfdx_v_fd = adjoint * res;
         x_pert.Add(-2 * delta, v);
         mesh->SetNodes(x_pert);
         res.Assemble();
         dfdx_v_fd -= adjoint * res;
         dfdx_v_fd /= (2 * delta);
         mesh->SetNodes(*x_nodes); // remember to reset the mesh nodes

         REQUIRE(dfdx_v == Approx(dfdx_v_fd).margin(1e-10));
      }
   }
}
*/

/** not maintaining anymore
TEST_CASE("ThermalSensIntegrator::AssembleElementVector",
          "[ThermalSensIntegrator]")
{
   using namespace mfem;
   using namespace euler_data;
   using namespace mach;

   const int dim = 3; // templating is hard here because mesh constructors
   double delta = 1e-5;

   // generate a 8 element mesh
   int num_edge = 2;
   std::unique_ptr<Mesh> mesh = electromag_data::getMesh();
                              //(new Mesh(num_edge, num_edge, num_edge, Element::TETRAHEDRON,
                              //        true, 1.0, 1.0, 1.0, true));
   mesh->EnsureNodes();

   for (int p = 1; p <= 4; ++p)
   {
      DYNAMIC_SECTION("...for degree p = " << p)
      {
         // get the finite-element space for the state and adjoint
         std::unique_ptr<FiniteElementCollection> fec(
             new H1_FECollection(p, dim));
         std::unique_ptr<FiniteElementCollection> feca(
             new ND_FECollection(p, dim));
         std::unique_ptr<FiniteElementSpace> fes(new FiniteElementSpace(
             mesh.get(), fec.get()));
         std::unique_ptr<FiniteElementSpace> fesa(new FiniteElementSpace(
             mesh.get(), feca.get()));

         // we use res for finite-difference approximation
         GridFunction A(fesa.get());
         VectorFunctionCoefficient perta(dim, electromag_data::randVectorState);
         A.ProjectCoefficient(perta);
         std::unique_ptr<Coefficient> q1(new ConstantCoefficient(1));
         std::unique_ptr<Coefficient> q2(new SteinmetzCoefficient(
                        1, 2, 4, 0.5, 0.6, A));
         std::unique_ptr<mach::MeshDependentCoefficient> Q;
         // Q.reset(new mach::MeshDependentCoefficient());
         // Q->addCoefficient(1, move(q1)); 
         // Q->addCoefficient(2, move(q2));
         std::unique_ptr<VectorCoefficient> QV(
            new SteinmetzVectorDiffCoefficient(1, 2, 4, 0.5, 0.6, A));
         LinearForm res(fes.get());
         res.AddDomainIntegrator(
            new DomainLFIntegrator(*q2));

         // initialize state and adjoint; here we randomly perturb a constant state
         GridFunction state(fes.get()), adjoint(fes.get());
         FunctionCoefficient pert(electromag_data::randState);
         state.ProjectCoefficient(pert);
         adjoint.ProjectCoefficient(pert);


         // build the nonlinear form for d(psi^T R)/dx 
         LinearForm dfdx_form(fesa.get());
         dfdx_form.AddDomainIntegrator(
            new mach::ThermalSensIntegrator(*QV, &adjoint));

         // initialize the vector that we use to perturb the vector potential
         GridFunction v(fesa.get());
         VectorFunctionCoefficient v_rand(dim, electromag_data::randVectorState);
         v.ProjectCoefficient(v_rand);

         // evaluate df/dx and contract with v
         //GridFunction dfdx(*x_nodes);
         dfdx_form.Assemble();
         double dfdx_v = dfdx_form * v;

         // now compute the finite-difference approximation...
         //GridFunction a_pert(A);
         A.Add(delta, v);
         res.Assemble();
         double dfdx_v_fd = adjoint * res;
         A.Add(-2 * delta, v);
         res.Assemble();
         dfdx_v_fd -= adjoint * res;
         dfdx_v_fd /= (2 * delta);
         A.Add(delta, v);

         REQUIRE(dfdx_v == Approx(dfdx_v_fd).margin(1e-10));
      }
   }
}
*/
