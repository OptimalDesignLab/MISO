#include <iostream>

#include "catch.hpp"
#include "mfem.hpp"
#include "nlohmann/json.hpp"

#include "mach.hpp"

// Provide the options explicitly for regression tests
auto options = R"(
{
   "space-dis": {
      "basis-type": "nedelec",
      "degree": 4
   },
   "components": {
      "winding": {
         "material": "copperwire",
         "attr": 1,
         "linear": true
      }
   }
})"_json;

/// \brief Exact solution for magnetic vector potential
/// \param[in] h - height of conducting slot
/// \param[in] b - width of conducting slot
/// \param[in] b0 - width of conducting slot opening
/// \param[in] x - coordinate of the point at which the state is needed
/// \param[out] A - magnetic vector potential
void aexact(double h, double b, double b0, double N, double Ipk,const Vector &x, Vector &A);

TEST_CASE("ACLossFunctionalIntegrator - Reddy Paper")
{
   using namespace mach;

   /// numbers from paper
   constexpr double h = 0.04;
   constexpr double b = 0.026875; // estimated from Figs 10-11
   constexpr double b0 = 0.002875; // estimated from Fig 2
   constexpr double N = 18;
   double Ipk = 125 * sqrt(2);
   
   std::unique_ptr<mfem::Mesh> mesh( new mfem::Mesh(
      // mfem::Mesh::MakeCartesian3D(32, 64, 1,
      //                             Element::TETRAHEDRON,
      //                             b, h, 1.0, true)));
      mfem::Mesh::MakeCartesian2D(16, 32,
                                  Element::QUADRILATERAL, true,
                                  b, h, true)));

   auto solver = createSolver<MagnetostaticSolver>(options, std::move(mesh));
   auto em_solver = dynamic_cast<MagnetostaticSolver*>(solver.get());

   auto state = solver->getNewField();
   solver->setFieldValue(*state, [&](const Vector &X, Vector &A)
   {
      aexact(h, b, b0, N, Ipk, X, A);
   });

   // solver->setField("state", *state);
   // solver->printField("state", "state");

   // auto B = solver->getField("B");
   // em_solver->calcCurl(*state, *B);
   // solver->setField("B", *B);
   // solver->printField("B", "B");

   solver->createOutput("ac_loss");

   std::vector<double> frequencies = {100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 1100, 1200, 1300, 1400, 1500, 2333};
   std::vector<double> loss;
   for (const auto &freq : frequencies)
   {
      MachInputs inputs {
         {"state", state->GetData()},
         {"strand_radius", 0.00081}, // 14 AWG
         {"frequency", freq}, // 14000 * 10 / 60}, // 14000 RPM, 10 poles
         {"num_strands", 10 * 18}, 
         {"slot_area", h * b},
         {"length", 0.075}
      };
      auto ac_loss = solver->calcOutput("ac_loss", inputs);
      loss.push_back(ac_loss * 12 * 2. / 3.);
   }

   std::cout << "frequencies: ";
   for (const auto &freq : frequencies)
   {
      std::cout << freq << ", ";
   }
   std::cout << "\n";

   std::cout << "ac loss: ";
   for (const auto &ac_loss : loss)
   {
      std::cout << ac_loss << ", ";
   }
   std::cout << "\n";

   auto wire_length = 2 * 0.075 * 18 * 12 / 3; // 2*stack_length * n_turns * n_slots / 3
   auto r_dc = wire_length / (M_PI * pow(0.00081, 2) * 10 * 58.14e6); // L / (copper_area * conductivity)
   auto dc_loss = 125 * 125 * r_dc * 0.5;

   std::cout << "dc loss: " << dc_loss << "\n";
}

void aexact(double h,
            double b,
            double b0,
            double N,
            double Ipk,
            const Vector &X,
            Vector &A)
{
   double mu0 = 4*M_PI * 1e-7;

   double k = M_PI / b;
   double H0 = N * Ipk / b;

   auto Hn = [&](int n)
   {
      return 4 * N * Ipk * sin(n * M_PI_2 * b0 / b) / (n * M_PI * b0);
   };

   auto AnCn = [&](int n)
   {
      return mu0 * Hn(n) / (n * k * (exp(n * k * h) - exp(-n * k * h)));
   };

   auto Az = [&](double x, double y) {
      double A_z = 0.5 * mu0 * H0 * y * y / h;

      for (int n = 2; n < 151; n += 2)
      {
         A_z += AnCn(n) * cos(n * k * x) * (exp(n * k * y) + exp(-n * k * y));
      }
      return A_z;
   };


   int dim = X.Size();
   int dimc = (dimc == 3) ? 3 : 1;
   A.SetSize(dimc);
   double x = X(0) - b/2;
   double y = X(1);

   if (dimc == 1)
   {
      A = Az(x, y);
   }
   else
   {
      A = 0.0;
      A(2) = Az(x, y);
   }
}