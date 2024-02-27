#include <random>
#include <vector>

#include "catch.hpp"
#include "mfem.hpp"
#include "nlohmann/json.hpp"

#include "electromag_test_data.hpp"
#include "reluctivity_coefficient.hpp"

namespace
{
// String used to define a single element mesh, with a c-shaped quad element
std::string mesh_str =
   "MFEM mesh v1.0"                    "\n\n"
   "dimension"                           "\n"
   "2"                                 "\n\n"
   "elements"                            "\n"
   "1"                                   "\n"
   "1 3 0 1 2 3"                       "\n\n"
   "boundary"                            "\n"
   "0"                                 "\n\n"
   "vertices"                            "\n"
   "4"                                 "\n\n"
   "nodes"                               "\n"
   "FiniteElementSpace"                  "\n"
   "FiniteElementCollection: Quadratic"  "\n"
   "VDim: 2"                             "\n"
   "Ordering: 1"                         "\n"
   "0 0"                                 "\n"
   "0 2"                                 "\n"
   "0 6"                                 "\n"
   "0 8"                                 "\n"
   "0 1"                                 "\n"
   "-6 4"                                "\n"
   "0 7"                                 "\n"
   "-8 4"                                "\n"
   "-7 4"                                "\n";

}  // anonymous namespace

TEST_CASE("logNuBBSplineReluctivityCoefficient::EvalStateDeriv")
{
   constexpr double eps_fd = 1e-5;
   std::default_random_engine generator;
   std::uniform_real_distribution<double> distribution(0.1,2.0);

   // Create quadratic mesh with single C-shaped quadrilateral
   std::stringstream meshStr;
   meshStr << mesh_str;
   mfem::Mesh mesh(meshStr);
   
   auto component = R"({
      "components": {
         "test": {
            "attrs": [1],
            "material": {
               "name": "hiperco50",
               "reluctivity": {
                  "model": "lognu",
                  "cps": [5.5286, 5.4645, 4.5597, 4.2891, 3.8445, 4.2880, 4.9505, 11.9364, 11.9738, 12.6554, 12.8097, 13.3347, 13.5871, 13.5871, 13.5871],
                  "knots": [0, 0, 0, 0, 0.1479, 0.5757, 0.9924, 1.4090, 1.8257, 2.2424, 2.6590, 3.0757, 3.4924, 3.9114, 8.0039, 10.0000, 10.0000, 10.0000, 10.0000],
                  "degree": 3
               }
            }
         }
      }
   })"_json;
   miso::ReluctivityCoefficient coeff(component, {});

   for (int p = 1; p <= 4; ++p)
   {
      const int dim = mesh.Dimension();
      mfem::H1_FECollection fec(p, dim);
      mfem::FiniteElementSpace fes(&mesh, &fec);

      const auto &el = *fes.GetFE(0);
      mfem::IsoparametricTransformation trans;
      mesh.GetElementTransformation(0, &trans);

      int order = 2 * el.GetOrder() - 2;
      const auto *ir = &mfem::IntRules.Get(el.GetGeomType(), order);

      for (int i = 0; i < ir->GetNPoints(); i++)
      {
         const auto &ip = ir->IntPoint(i);
         trans.SetIntPoint(&ip);

         double state = distribution(generator);
         double deriv = coeff.EvalStateDeriv(trans, ip, state);

         state += eps_fd;
         double deriv_fd_p = coeff.Eval(trans, ip, state);
         state -= 2.0*eps_fd;
         double deriv_fd_m = coeff.Eval(trans, ip, state);

         double deriv_fd = (deriv_fd_p - deriv_fd_m) / (2.0 * eps_fd);
         REQUIRE(deriv == Approx(deriv_fd));
      }
   }

}

TEST_CASE("logNuBBSplineReluctivityCoefficient::EvalState2ndDeriv")
{
   constexpr double eps_fd = 1e-5;
   std::default_random_engine generator;
   std::uniform_real_distribution<double> distribution(0.1,2.0);

   // Create quadratic mesh with single C-shaped quadrilateral
   std::stringstream meshStr;
   meshStr << mesh_str;
   mfem::Mesh mesh(meshStr);
   
   auto component = R"({
      "components": {
         "test": {
            "attrs": [1],
            "material": {
               "name": "hiperco50",
               "reluctivity": {
                  "model": "lognu",
                  "cps": [5.5286, 5.4645, 4.5597, 4.2891, 3.8445, 4.2880, 4.9505, 11.9364, 11.9738, 12.6554, 12.8097, 13.3347, 13.5871, 13.5871, 13.5871],
                  "knots": [0, 0, 0, 0, 0.1479, 0.5757, 0.9924, 1.4090, 1.8257, 2.2424, 2.6590, 3.0757, 3.4924, 3.9114, 8.0039, 10.0000, 10.0000, 10.0000, 10.0000],
                  "degree": 3
               }
            }
         }
      }
   })"_json;
   miso::ReluctivityCoefficient coeff(component, {});

   for (int p = 1; p <= 4; ++p)
   {
      const int dim = mesh.Dimension();
      mfem::H1_FECollection fec(p, dim);
      mfem::FiniteElementSpace fes(&mesh, &fec);

      const auto &el = *fes.GetFE(0);
      mfem::IsoparametricTransformation trans;
      mesh.GetElementTransformation(0, &trans);

      int order = 2 * el.GetOrder() - 2;
      const auto *ir = &mfem::IntRules.Get(el.GetGeomType(), order);

      for (int i = 0; i < ir->GetNPoints(); i++)
      {
         const auto &ip = ir->IntPoint(i);
         trans.SetIntPoint(&ip);

         double state = distribution(generator);
         double second_deriv = coeff.EvalState2ndDeriv(trans, ip, state);

         state += eps_fd;
         double second_deriv_fd_p = coeff.EvalStateDeriv(trans, ip, state);
         state -= 2.0*eps_fd;
         double second_deriv_fd_m = coeff.EvalStateDeriv(trans, ip, state);

         double second_deriv_fd = (second_deriv_fd_p - second_deriv_fd_m) / (2.0 * eps_fd);
         REQUIRE(second_deriv == Approx(second_deriv_fd));
      }
   }
}