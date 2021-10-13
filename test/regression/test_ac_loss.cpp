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
      //                             b, h, 0.001, true)));
      mfem::Mesh::MakeCartesian2D(16, 32,
                                  Element::TRIANGLE, true,
                                  b, h, true)));

   auto solver = createSolver<MagnetostaticSolver>(options, std::move(mesh));
   auto em_solver = dynamic_cast<MagnetostaticSolver*>(solver.get());

   auto state = solver->getNewField();
   solver->setFieldValue(*state, [&](const Vector &X, Vector &A)
   {
      aexact(h, b, b0, N, Ipk, X, A);
   });

   solver->setField("state", *state);
   solver->printField("state", "state");

   auto B = solver->getField("B");
   em_solver->calcCurl(*state, *B);
   solver->setField("B", *B);
   solver->printField("B_new", "B");

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
      std::cout << ac_loss * 1000 << ", ";
   }
   std::cout << "\n";

   auto wire_length = 2 * 0.075 * 18 * 12 / 3; // 2*stack_length * n_turns * n_slots / 3
   auto r_dc = wire_length / (M_PI * pow(0.00081, 2) * 10 * 58.14e6); // L / (copper_area * conductivity)
   auto dc_loss = 125 * 125 * r_dc * 0.5;

   std::cout << "dc loss: " << dc_loss << "\n";
}

/// 2D: 0.420313, 1.68125, 3.78282, 6.72501, 10.5078, 15.1313, 20.5953, 26.9, 34.0454, 42.0313, 50.8579, 60.5251, 71.0329, 82.3814, 94.5705, 228.772
/// 3D: 0.421145, 1.68458, 3.7903, 6.73831, 10.5286, 15.1612, 20.6361, 26.9533, 34.1127, 42.1145, 50.9585, 60.6448, 71.1734, 82.5443, 94.7575, 229.224
/// 3D: 0.0421145, 0.168458, 0.37903, 0.673831, 1.05286, 1.51612, 2.06361, 2.69533, 3.41127, 4.21145, 5.09585, 6.06448, 7.11734, 8.25443, 9.47575, 22.9224 // 0.1
/// 3D: 0.00421146, 0.0168458, 0.0379032, 0.0673834, 0.105287, 0.151613, 0.206362, 0.269534, 0.341128, 0.421146, 0.509587, 0.606451, 0.711737, 0.825447, 0.947579, 2.29225 // 0.01
/// 3D: 0.000421304, 0.00168521, 0.00379173, 0.00674086, 0.0105326, 0.0151669, 0.0206439, 0.0269634, 0.0341256, 0.0421304, 0.0509777, 0.0606677, 0.0712003, 0.0825755, 0.0947933, 0.229311 // 0.001
/// 3D: 0.421304, 1.68521, 3.79173, 6.74086, 10.5326, 15.1669, 20.6439, 26.9634, 34.1256, 42.1304, 50.9777, 60.6677, 71.2003, 82.5755, 94.7933, 229.311

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