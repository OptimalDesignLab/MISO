#include <random>
#include <vector>

#include "catch.hpp"
#include "mfem.hpp"
#include "nlohmann/json.hpp"

#include "coefficient.hpp"
#include "material_library.hpp"

#include "electromag_test_data.hpp"

#include "reluctivity_coefficient.hpp"
#include "conductivity_coefficient.hpp"
#include "cal2_kh_coefficient.hpp"
#include "cal2_ke_coefficient.hpp"
#include "pm_demag_constraint_coeff.hpp"
#include "remnant_flux_coefficient.hpp"

namespace
{

using namespace mfem;

double func(const Vector &x)
{
  return (x(0) + x(1) + x(2));
}

void funcRevDiff(const Vector &x, const double Q_bar, Vector &x_bar)
{
   x_bar.SetSize(3);
   x_bar = Q_bar;
}

void vectorFunc(const Vector &x, Vector &y)
{
   y.SetSize(3);
   y(0) = x(0)*x(0) - x(1);
   y(1) = x(0) * exp(x(1));
   y(2) = x(2)*x(0) - x(1);
}

void vectorFuncRevDiff(const Vector &x, const Vector &v_bar, Vector &x_bar)
{
   x_bar(0) = v_bar(0) * 2*x(0) + v_bar(1) * exp(x(1)) + v_bar(2)*x(2);
   x_bar(1) = -v_bar(0) + v_bar(1) * x(0) * exp(x(1)) - v_bar(2); 
   x_bar(2) = v_bar(2) * x(0); 
}

void vectorFunc2(const Vector &x, Vector &y)
{
   y.SetSize(3);
   y(0) = sin(x(0))*x(2)*x(2);
   y(1) = x(1) - x(0)*x(2);
   y(2) = sin(x(1))*exp(x(2));
}

void vectorFunc2RevDiff(const Vector &x, const Vector &v_bar, Vector &x_bar)
{
   x_bar(0) = cos(x(0))*x(2)*x(2)*v_bar(0) - x(2)*v_bar(1);
   x_bar(1) = v_bar(1) + cos(x(1))*exp(x(2))*v_bar(2); 
   x_bar(2) = 2*sin(x(0))*x(2)*v_bar(0) - x(0)*v_bar(1) + sin(x(1))*exp(x(2))*v_bar(2);
}

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


std::string one_tet_mesh_str = 
   "MFEM mesh v1.0\n\n"
   "dimension\n"
   "3\n\n"
   "elements\n"
   "1\n"
   "1 4 0 1 2 3\n\n"
   "boundary\n"
   "4\n"
   "1 2 0 1 2\n"
   "2 2 0 1 3\n"
   "3 2 1 2 3\n"
   "4 2 0 2 3\n\n"
   "vertices\n"
   "4\n\n"
   "nodes\n"
   "FiniteElementSpace\n"
   "FiniteElementCollection: H1_3D_P1\n"
   "VDim: 3\n"
   "Ordering: 1\n\n"
   "0 0 0\n"
   "1 0 0\n"
   "0 1 0\n"
   "0 0 1\n";

std::string two_tet_mesh_str = 
   "MFEM mesh v1.0\n\n"
   "dimension\n"
   "3\n\n"
   "elements\n"
   "2\n"
   "1 4 0 1 2 3\n"
   "2 4 1 2 3 4\n\n"
   "boundary\n"
   "7\n"
   "1 2 0 1 2\n"
   "2 2 0 1 3\n"
   "3 2 1 2 3\n"
   "4 2 0 2 3\n"
   "5 2 1 3 4\n"
   "6 2 1 2 4\n"
   "7 2 2 3 4\n\n"
   "vertices\n"
   "5\n\n"
   "nodes\n"
   "FiniteElementSpace\n"
   "FiniteElementCollection: H1_3D_P1\n"
   "VDim: 3\n"
   "Ordering: 1\n\n"
   "0 0 0\n"
   "1 0 0\n"
   "0 1 0\n"
   "0 0 1\n"
   "1 0 1\n";

std::default_random_engine gener;
std::uniform_real_distribution<double> uniform(-1.0,1.0);

}


TEST_CASE("MeshDependentVectorCoefficient::EvalRevDiff",
          "[MeshDependentVectorCoefficient]")
{
   using namespace mfem;
   // using namespace electromag_data;
   using namespace miso;

   constexpr double eps_fd = 1e-5;
   constexpr int dim = 3;

   std::stringstream meshStr;
   meshStr << two_tet_mesh_str;
   Mesh mesh(meshStr);

   /// Costruct coefficient
   VectorMeshDependentCoefficient coeff(dim);

   std::unique_ptr<mfem::VectorCoefficient> coeff1(
      new VectorFunctionCoefficient(dim, vectorFunc, vectorFuncRevDiff));
   std::unique_ptr<mfem::VectorCoefficient> coeff2(
      new VectorFunctionCoefficient(dim, vectorFunc2, vectorFunc2RevDiff));
   
   coeff.addCoefficient(1, move(coeff1));
   coeff.addCoefficient(2, move(coeff2));

   for (int p = 1; p <= 1; p++)
   {
      /// construct elements
      ND_FECollection fec(p, dim);
      FiniteElementSpace fes(&mesh, &fec);

      for (int j = 0; j < fes.GetNE(); j++)
      {

         const FiniteElement &el = *fes.GetFE(j);

         IsoparametricTransformation trans;
         mesh.GetElementTransformation(j, &trans);

         const IntegrationRule *ir = NULL;
         {
            int order = trans.OrderW() + 2 * el.GetOrder();
            ir = &IntRules.Get(el.GetGeomType(), order);
         }

         DenseMatrix &coords = trans.GetPointMat();
         DenseMatrix coords_bar(coords.Height(), coords.Width());

         // V_bar is the vector contrated with the derivative of the projection
         // the values are not important for this test
         Vector V_bar(dim);

         for (int i = 0; i < ir->GetNPoints(); i++)
         {
            for (int v = 0; v < V_bar.Size(); ++v)
            {
               V_bar(v) = uniform(gener);
            }

            const IntegrationPoint &ip = ir->IntPoint(i);

            trans.SetIntPoint(&ip);

            // reverse-mode differentiation of eval
            coords_bar = 0.0;
            coeff.EvalRevDiff(V_bar, trans, ip, coords_bar);

            // get the weighted derivatives using finite difference method
            for (int n = 0; n < coords.Width(); ++n)
            {
               for (int di = 0; di < coords.Height(); ++di)
               {

                  Vector vb(dim), vf(dim);

                  coords(di, n) += eps_fd;
                  coeff.Eval(vf, trans, ip);
                  coords(di, n) -= 2.0*eps_fd;
                  coeff.Eval(vb, trans, ip);

                  vf -= vb;
                  vf *= 1.0/(2.0*eps_fd);
                  coords(di, n) += eps_fd;
                  double v_bar_fd = V_bar * vf;

                  REQUIRE(coords_bar(di, n) == Approx(v_bar_fd));
               }
            }
         }
      }
   }
}

TEST_CASE("ScalarVectorProductCoefficient::EvalStateDeriv")
{
   
}

TEST_CASE("ScalarVectorProductCoefficient::EvalRevDiff")
{
   using namespace mfem;
   using namespace miso;

   constexpr double eps_fd = 1e-5;
   constexpr int dim = 3;

   // Create quadratic mesh with single C-shaped quadrilateral
   std::stringstream meshStr;
   meshStr << one_tet_mesh_str;
   Mesh mesh(meshStr);

   FunctionCoefficient scalar(func, funcRevDiff);
   VectorFunctionCoefficient vec(dim, vectorFunc, vectorFuncRevDiff);

   /// Costruct coefficient
   miso::ScalarVectorProductCoefficient coeff(scalar, vec);

   for (int p = 1; p <= 4; p++)
   {
      /// construct elements
      H1_FECollection fec(p, dim);
      FiniteElementSpace fes(&mesh, &fec);

      for (int j = 0; j < fes.GetNE(); j++)
      {

         const FiniteElement &el = *fes.GetFE(j);

         IsoparametricTransformation trans;
         mesh.GetElementTransformation(j, &trans);

         const IntegrationRule *ir = NULL;
         {
            int order = trans.OrderW() + 2 * el.GetOrder();
            ir = &IntRules.Get(el.GetGeomType(), order);
         }

         DenseMatrix &coords = trans.GetPointMat();
         DenseMatrix coords_bar(coords.Height(), coords.Width());

         // V_bar is the vector contrated with the derivative of the projection
         // the values are not important for this test
         Vector V_bar(dim);

         for (int i = 0; i < ir->GetNPoints(); i++)
         {
            for (int v = 0; v < V_bar.Size(); ++v)
            {
               V_bar(v) = uniform(gener);
            }

            const IntegrationPoint &ip = ir->IntPoint(i);

            trans.SetIntPoint(&ip);

            // reverse-mode differentiation of eval
            coords_bar = 0.0;
            coeff.EvalRevDiff(V_bar, trans, ip, 0.0, coords_bar);

            // get the weighted derivatives using finite difference method
            for (int n = 0; n < coords.Width(); ++n)
            {
               for (int di = 0; di < coords.Height(); ++di)
               {

                  Vector vb(dim), vf(dim);

                  coords(di, n) += eps_fd;
                  coeff.Eval(vf, trans, ip);
                  coords(di, n) -= 2.0*eps_fd;
                  coeff.Eval(vb, trans, ip);

                  vf -= vb;
                  vf *= 1.0/(2.0*eps_fd);
                  coords(di, n) += eps_fd;
                  double v_bar_fd = V_bar * vf;

                  REQUIRE(coords_bar(di, n) == Approx(v_bar_fd));
               }
            }
         }
      }
   }
}

TEST_CASE("FunctionCoefficient::EvalRevDiff",
          "[FunctionCoefficient]")
{
   using namespace mfem;
   // using namespace electromag_data;
   using namespace miso;

   constexpr double eps_fd = 1e-5;
   constexpr int dim = 3;

   std::stringstream meshStr;
   meshStr << two_tet_mesh_str;
   Mesh mesh(meshStr);

   /// Costruct coefficient
   FunctionCoefficient coeff(func, funcRevDiff);

   for (int p = 1; p <= 1; p++)
   {
      /// construct elements
      H1_FECollection fec(p, dim);
      FiniteElementSpace fes(&mesh, &fec);

      for (int j = 0; j < fes.GetNE(); j++)
      {

         const FiniteElement &el = *fes.GetFE(j);

         IsoparametricTransformation trans;
         mesh.GetElementTransformation(j, &trans);

         const IntegrationRule *ir = NULL;
         {
            int order = trans.OrderW() + 2 * el.GetOrder();
            ir = &IntRules.Get(el.GetGeomType(), order);
         }

         DenseMatrix &coords = trans.GetPointMat();
         DenseMatrix coords_bar(coords.Height(), coords.Width());

         // V_bar is the vector contrated with the derivative of the projection
         // the values are not important for this test
         double Q_bar;

         for (int i = 0; i < ir->GetNPoints(); i++)
         {

            Q_bar = uniform(gener);

            const IntegrationPoint &ip = ir->IntPoint(i);

            trans.SetIntPoint(&ip);

            // reverse-mode differentiation of eval
            coords_bar = 0.0;
            coeff.EvalRevDiff(Q_bar, trans, ip, coords_bar);

            // get the weighted derivatives using finite difference method
            for (int n = 0; n < coords.Width(); ++n)
            {
               for (int di = 0; di < coords.Height(); ++di)
               {
                  coords(di, n) += eps_fd;
                  double vf = coeff.Eval(trans, ip);
                  coords(di, n) -= 2.0*eps_fd;
                  vf -= coeff.Eval(trans, ip);

                  vf *= 1.0/(2.0*eps_fd);
                  coords(di, n) += eps_fd;
                  double v_bar_fd = Q_bar * vf;

                  REQUIRE(coords_bar(di, n) == Approx(v_bar_fd));
               }
            }
         }
      }
   }
}

/** not maintaining anymore
TEST_CASE("SteinmetzCoefficient::EvalRevDiff",
          "[SteinmetzCoefficient]")
{
   using namespace mfem;
   // using namespace electromag_data;
   using namespace miso;

   constexpr double eps_fd = 1e-5;
   constexpr int dim = 3;

   // std::stringstream meshStr;
   // meshStr << two_tet_mesh_str;
   // Mesh mesh(meshStr);

   Mesh mesh(1, 2, 2, Element::TETRAHEDRON,
             true, 1.0, 1.0, 1.0, true);

   /// Costruct coefficient
   for (int p = 1; p <= 1; p++)
   {
      /// construct elements
      ND_FECollection fec(p, dim);
      FiniteElementSpace fes(&mesh, &fec);

      GridFunction A(&fes);
      VectorFunctionCoefficient pert(dim, electromag_data::randVectorState);
      A.ProjectCoefficient(pert);

      miso::SteinmetzCoefficient coeff(1, 2, 4, 0.5, 0.6, A);

      for (int j = 0; j < fes.GetNE(); j++)
      {
         const FiniteElement &el = *fes.GetFE(j);

         IsoparametricTransformation trans;
         mesh.GetElementTransformation(j, &trans);

         const IntegrationRule *ir = NULL;
         {
            int order = trans.OrderW() + 2 * el.GetOrder();
            ir = &IntRules.Get(el.GetGeomType(), order);
         }

         DenseMatrix &coords = trans.GetPointMat();
         DenseMatrix coords_bar(coords.Height(), coords.Width());

         for (int i = 0; i < ir->GetNPoints(); i++)
         {
            // Q_bar is the number multiplied by the derivative of the
            // projection, the values are not important
            double Q_bar = uniform(gener);
            const IntegrationPoint &ip = ir->IntPoint(i);

            trans.SetIntPoint(&ip);

            // reverse-mode differentiation of eval
            coords_bar = 0.0;
            coeff.EvalRevDiff(Q_bar, trans, ip, coords_bar);

            // get the weighted derivatives using finite difference method
            for (int n = 0; n < coords.Width(); ++n)
            {
               for (int di = 0; di < coords.Height(); ++di)
               {
                  coords(di, n) += eps_fd;
                  double vf = coeff.Eval(trans, ip);
                  coords(di, n) -= 2.0*eps_fd;
                  vf -= coeff.Eval(trans, ip);

                  vf *= 1.0/(2.0*eps_fd);
                  coords(di, n) += eps_fd;
                  double q_bar_fd = Q_bar * vf;

                  REQUIRE(coords_bar(di, n) == Approx(q_bar_fd));
               }
            }
         }
      }
   }
}
*/

/** not maintaining anymore
TEST_CASE("SteinmetzVectorDiffCoefficient::Eval",
          "[SteinmetzVectorDiffCoefficient]")
{
   using namespace mfem;
   using namespace miso;

   constexpr double eps_fd = 1e-5;
   constexpr int dim = 3;

   std::stringstream meshStr;
   meshStr << two_tet_mesh_str;
   Mesh mesh(meshStr);

   /// Costruct coefficient
   for (int p = 1; p <= 1; p++)
   {
      /// construct elements
      ND_FECollection fec(p, dim);
      FiniteElementSpace fes(&mesh, &fec);

      GridFunction A(&fes);
      VectorFunctionCoefficient pert(dim, electromag_data::randVectorState);
      A.ProjectCoefficient(pert);

      miso::SteinmetzCoefficient coeff(1, 2, 4, 0.5, 0.6, A);
      miso::SteinmetzVectorDiffCoefficient d_coeff(1, 2, 4, 0.5, 0.6, A);

      for (int j = 0; j < fes.GetNE(); j++)
      {

         const FiniteElement &el = *fes.GetFE(j);

         IsoparametricTransformation trans;
         mesh.GetElementTransformation(j, &trans);

         const IntegrationRule *ir = NULL;
         {
            int order = trans.OrderW() + 2 * el.GetOrder();
            ir = &IntRules.Get(el.GetGeomType(), order);
         }

         Vector A_bar(el.GetDof());

         for (int i = 0; i < ir->GetNPoints(); i++)
         {
            const IntegrationPoint &ip = ir->IntPoint(i);

            trans.SetIntPoint(&ip);
            Array<int> vdofs;
            Vector elfun;
            A.FESpace()->GetElementVDofs(trans.ElementNo, vdofs);
            A.GetSubVector(vdofs, elfun);

            Vector v(elfun.Size());
            for (int k = 0; k < elfun.Size(); k++)
            {
               v(k) = uniform(gener);
            }

            // reverse-mode differentiation of eval
            A_bar = 0.0;
            d_coeff.Eval(A_bar, trans, ip);
            double A_bar_v = A_bar * v;

            // get the weighted derivatives using finite difference method
            elfun.Add(eps_fd, v);
            A.SetSubVector(vdofs, elfun);
            double vf = coeff.Eval(trans, ip);
            elfun.Add(-2.0*eps_fd, v);
            A.SetSubVector(vdofs, elfun);
            vf -= coeff.Eval(trans, ip);

            vf *= 1.0/(2.0*eps_fd);
            elfun.Add(eps_fd, v);
            A.SetSubVector(vdofs, elfun);

            double q_bar_fd = vf;

            REQUIRE(A_bar_v == Approx(q_bar_fd));
         }
      }
   }
}
*/

void printVector(const std::vector<double> &vector)
{
   for (int k = 0; k < vector.size(); ++k)
   {
      std::cout << vector[k] << ", ";
   }
}

TEST_CASE("ReluctivityCoefficient lognu vs bh")
{
   using namespace mfem;
   using namespace miso;

   std::stringstream meshStr;
   meshStr << two_tet_mesh_str;
   Mesh mesh(meshStr);

   const int dim = mesh.SpaceDimension();


   /// Costruct coefficient
   for (int p = 1; p <= 1; p++)
   {
      /// construct elements
      ND_FECollection fec(p, dim);
      FiniteElementSpace fes(&mesh, &fec);

                     // "cps": [5.4094, 4.5222, 3.8259, 3.7284, 5.1554, 11.1488, 13.0221, 13.5798, 13.5808, 13.5814],

      const auto &lognu_options = R"(
      {
         "components": {
            "test": {
               "attrs": 1,
               "material": {
                  "name": "hiperco50",
                  "reluctivity": {
                     "model": "lognu",
                     "cps": [5.40954787023781, 4.50729991768494, 3.82622972028510, 3.73337170624446, 5.16008222491274, 11.0710865890706, 12.6733270435251, 13.5870714039890, 13.5870714039890, 13.5870714039890],
                     "knots": [0, 0, 0, 0, 0.754565031471487, 1.71725877985567, 2.14583020842710, 2.57440163699853, 3.00297306556996, 3.56974470521025, 6, 6, 6, 6],
                     "degree": 3
                  }
               }
            }
         }
      })"_json;
      auto lognu_coeff = ReluctivityCoefficient(lognu_options, material_library);

      const auto &bh_options = R"(
      {
         "components": {
            "test": {
               "attrs": 1,
               "material": {
                  "name": "hiperco50",
                  "reluctivity": {
                     "model": "bh"
                  }
               }
            }
         }
      })"_json;      
      auto bh_coeff = ReluctivityCoefficient(bh_options, material_library);

      int npts = 1000;
      std::vector<double> b_mags(npts);
      double b_max = 10.0;
      for (int i = 0; i < npts; ++i)
      {
         b_mags[i] = double(i) / double(npts) * b_max;
      }

      std::vector<double> lognu_nu(npts);
      std::vector<double> lognu_dnudb(npts);

      std::vector<double> bh_nu(npts);
      std::vector<double> bh_dnudb(npts);

      // for (int j = 0; j < fes.GetNE(); j++)
      for (int j = 0; j < 1; j++)
      {

         const FiniteElement &el = *fes.GetFE(j);

         IsoparametricTransformation trans;
         mesh.GetElementTransformation(j, &trans);

         const IntegrationRule *ir = NULL;
         {
            int order = trans.OrderW() + 2 * el.GetOrder();
            ir = &IntRules.Get(el.GetGeomType(), order);
         }

         // for (int i = 0; i < ir->GetNPoints(); i++)
         for (int i = 0; i < 1; i++)
         {
            const IntegrationPoint &ip = ir->IntPoint(i);

            trans.SetIntPoint(&ip);

            for (int k = 0; k < npts; ++k)
            {
               auto b_mag = b_mags[k];

               lognu_nu[k] = lognu_coeff.Eval(trans, ip, b_mag);
               lognu_dnudb[k] = lognu_coeff.EvalStateDeriv(trans, ip, b_mag);
               bh_nu[k] = bh_coeff.Eval(trans, ip, b_mag);
               bh_dnudb[k] = bh_coeff.EvalStateDeriv(trans, ip, b_mag);
            }

            /// Whenever want to see output, uncomment this section
            /*
            std::cout << "b = np.array([";
            printVector(b_mags);
            std::cout << "])\n";

            std::cout << "lognu_nu = np.array([";
            printVector(lognu_nu);
            std::cout << "])\n";
            std::cout << "lognu_dnudb = np.array([";
            printVector(lognu_dnudb);
            std::cout << "])\n";

            std::cout << "bh_nu = np.array([";
            printVector(bh_nu);
            std::cout << "])\n";
            std::cout << "bh_dnudb = np.array([np.";
            printVector(bh_dnudb);
            std::cout << "])\n";
            */
         }
      }
   }
}

TEST_CASE("ConductivityCoefficient: Models vs. Desired")
{
   using namespace mfem;
   using namespace miso;

   std::stringstream meshStr;
   meshStr << two_tet_mesh_str;
   Mesh mesh(meshStr);

   const int dim = mesh.SpaceDimension();


   /// Construct coefficient
   for (int p = 1; p <= 1; p++)
   {
      /// construct elements
      ND_FECollection fec(p, dim);
      FiniteElementSpace fes(&mesh, &fec);

      const auto &TempDepSigma_options = R"(
      {
         "components": {
            "test": {
               "attrs": 1,
               "material": {
                  "name": "hiperco50",
                  "conductivity": {
                     "model": "linear",
                     "sigma_T_ref": 5.6497e7,
                     "T_ref": 293.15,
                     "alpha_resistivity": 3.8e-3
                  }
               }
            }
         }
      })"_json;
      auto TempDepSigma_coeff = ConductivityCoefficient(TempDepSigma_options, material_library);

      const auto &ConstantSigma_options = R"(
      {
         "components": {
            "test": {
               "attrs": 1,
               "material": {
                  "name": "hiperco50",
                  "conductivity": {
                     "model": "constant"
                  }
               }
            }
         }
      })"_json;
      auto ConstantSigma_coeff = ConductivityCoefficient(ConstantSigma_options, material_library);

      const auto &OldSigma_options = R"(
      {
         "components": {
            "test": {
               "attrs": 1,
               "material": {
                  "name": "hiperco50",
                  "sigma": 58.14e6
               }
            }
         }
      })"_json;
      auto OldSigma_coeff = ConductivityCoefficient(OldSigma_options, material_library);

      int npts = 201;
      std::vector<double> temperatures(npts);
      for (int i = 0; i < npts; ++i)
      {
         temperatures[i] = double(i);
      }

      std::vector<double> TempDepSigma_sigma(npts);
      std::vector<double> TempDepSigma_dsigmadT(npts);

      std::vector<double> ConstantSigma_sigma(npts);
      std::vector<double> ConstantSigma_dsigmadT(npts);

      std::vector<double> OldSigma_sigma(npts);
      std::vector<double> OldSigma_dsigmadT(npts);

      // for (int j = 0; j < fes.GetNE(); j++)
      for (int j = 0; j < 1; j++)
      {

         const FiniteElement &el = *fes.GetFE(j);

         IsoparametricTransformation trans;
         mesh.GetElementTransformation(j, &trans);

         const IntegrationRule *ir = NULL;
         {
            int order = trans.OrderW() + 2 * el.GetOrder();
            ir = &IntRules.Get(el.GetGeomType(), order);
         }

         // for (int i = 0; i < ir->GetNPoints(); i++)
         for (int i = 0; i < 1; i++)
         {
            const IntegrationPoint &ip = ir->IntPoint(i);

            trans.SetIntPoint(&ip);

            for (int k = 0; k < npts; ++k)
            {
               auto temperature = temperatures[k];

               TempDepSigma_sigma[k] = TempDepSigma_coeff.Eval(trans, ip, temperature);
               TempDepSigma_dsigmadT[k] = TempDepSigma_coeff.EvalStateDeriv(trans, ip, temperature);
               ConstantSigma_sigma[k] = ConstantSigma_coeff.Eval(trans, ip, temperature);
               ConstantSigma_dsigmadT[k] = ConstantSigma_coeff.EvalStateDeriv(trans, ip, temperature);
               OldSigma_sigma[k] = OldSigma_coeff.Eval(trans, ip, temperature);
               OldSigma_dsigmadT[k] = OldSigma_coeff.EvalStateDeriv(trans, ip, temperature);
            }

            /// Whenever want to see output, uncomment this section
            /*
            std::cout << "temperatures = np.array([";
            printVector(temperatures);
            std::cout << "])\n";

            std::cout << "TempDepSigma_sigma = np.array([";
            printVector(TempDepSigma_sigma);
            std::cout << "])\n";
            std::cout << "TempDepSigma_dsigmadT = np.array([";
            printVector(TempDepSigma_dsigmadT);
            std::cout << "])\n";

            std::cout << "ConstantSigma_sigma = np.array([";
            printVector(ConstantSigma_sigma);
            std::cout << "])\n";
            std::cout << "ConstantSigma_dsigmadT = np.array([";
            printVector(ConstantSigma_dsigmadT);
            std::cout << "])\n";

            std::cout << "OldSigma_sigma = np.array([";
            printVector(OldSigma_sigma);
            std::cout << "])\n";
            std::cout << "OldSigma_dsigmadT = np.array([";
            printVector(OldSigma_dsigmadT);
            std::cout << "])\n";
            */
         }
      }
   }
}

///Adding in low-level Steinmetz test to make sure retrieves parameters correctly from new Core Loss JSON structure
///NOTE: Started this test, but don't believe it is necessary. Tested with a simple cout in coefficient.cpp shows deals with the new structure fine.
// TEST_CASE("Steinmetz Coefficients with new Core Loss JSON Structure")
// {
//    using namespace mfem;
//    using namespace miso;

//    std::stringstream meshStr;
//    meshStr << two_tet_mesh_str;
//    Mesh mesh(meshStr);

//    const int dim = mesh.SpaceDimension();
   
//    /// Construct Steinmetz coefficients
//    for (int p = 1; p <= 1; p++)
//    {
//       /// construct elements
//       ND_FECollection fec(p, dim);
//       FiniteElementSpace fes(&mesh, &fec);

//       ///TODO: Ensure these options are being used (rather than material library)
//       const auto &Steinmetz_options = R"(
//       {
//          "components": {
//             "test": {
//                "attrs": 1,
//                "material": {
//                   "name": "hiperco50",
//                   "core_loss": {
//                      "model": "steinmetz",
//                      "ks": 0.0044,
//                      "alpha": 1.286,
//                      "beta": 1.76835
//                   }
//                }
//             }
//          }
//       })"_json;

//       // auto rho = std::make_unique<miso::MeshDependentCoefficient>(constructMaterialCoefficient("rho", Steinmetz_options["components"], Steinmetz_options["components"]["materials"]));
//       // auto k_s = std::make_unique<miso::MeshDependentCoefficient>(constructMaterialCoefficient("ks", Steinmetz_options["components"], Steinmetz_options["components"]["materials"]));
//       // auto alpha = std::make_unique<miso::MeshDependentCoefficient>(constructMaterialCoefficient("alpha", Steinmetz_options["components"], Steinmetz_options["components"]["materials"]));
//       // auto beta = std::make_unique<miso::MeshDependentCoefficient>(constructMaterialCoefficient("beta", Steinmetz_options["components"], Steinmetz_options["components"]["materials"]));

//       auto CAL2_kh_coeff = CAL2khCoefficient(CAL2_options, material_library);
//       auto CAL2_ke_coeff = CAL2keCoefficient(CAL2_options, material_library);

//       // for (int j = 0; j < fes.GetNE(); j++)
//       for (int j = 0; j < 1; j++)
//       {

//          const FiniteElement &el = *fes.GetFE(j);

//          IsoparametricTransformation trans;
//          mesh.GetElementTransformation(j, &trans);

//          const IntegrationRule *ir = NULL;
//          {
//             int order = trans.OrderW() + 2 * el.GetOrder();
//             ir = &IntRules.Get(el.GetGeomType(), order);
//          }

//          // for (int i = 0; i < ir->GetNPoints(); i++)
//          for (int i = 0; i < 1; i++)
//          {
//             const IntegrationPoint &ip = ir->IntPoint(i);

//             trans.SetIntPoint(&ip);
            
//             auto rho = rho.Eval(trans, ip);
//             double k_s = k_s.Eval(trans, ip);
//             double alpha = alpha.Eval(trans, ip);
//             double beta = beta.Eval(trans, ip);

//             std::cout << "rho = " << rho << "\n";
//             std::cout << "k_s = " << k_s << "\n";
//             std::cout << "alpha = " << alpha << "\n";
//             std::cout << "beta = " << beta << "\n";
//          }
//       }
//    }
// }

TEST_CASE("CAL2 Coefficient: Models vs. Desired")
{
   using namespace mfem;
   using namespace miso;

   std::stringstream meshStr;
   meshStr << two_tet_mesh_str;
   Mesh mesh(meshStr);

   const int dim = mesh.SpaceDimension();


   /// Construct coefficient
   for (int p = 1; p <= 1; p++)
   {
      /// construct elements
      ND_FECollection fec(p, dim);
      FiniteElementSpace fes(&mesh, &fec);

      // Set the options in JSON format for the CAL2 coefficients
      const auto &CAL2_options = R"(
      {
         "components": {
            "test": {
               "attrs": 1,
               "material": {
                  "name": "hiperco50",
                  "core_loss": {
                     "model": "CAL2",
                     "T0": 293.15,
                     "kh_T0": [1.0e-02, 2.0e-02, 3.0e-02, 4.0e-02],
                     "ke_T0": [1.0e-07, 1.0e-06, 1.0e-05, 1.0e-04],
                     "T1": 473.15,
                     "kh_T1": [3.5e-02, 2.5e-02, 1.5e-02, 0.5e-02],
                     "ke_T1": [1.0e-04, 1.0e-05, 1.0e-06, 1.0e-07]
                  }
               }
            }
         }
      })"_json;

      auto CAL2_kh_coeff = CAL2khCoefficient(CAL2_options, material_library);
      auto CAL2_ke_coeff = CAL2keCoefficient(CAL2_options, material_library);

      int npts = 50;
      std::vector<double> temperatures(npts);
      std::vector<double> max_fluxes(npts);
      double frequency = 1000;
      // Note, these cal2 coeffients are independent of frequency
      // Temperature spans between 0 and about 200 degrees
      // Max fluxes span between 0 and about 2 T
      for (int i = 0; i < npts; ++i)
      {
         temperatures[i] = double(i)*(200/double(npts));
         max_fluxes[i] = 2*(double(i)/8-floor(double(i)/8));
      }

      std::vector<double> CAL2_kh(npts);
      std::vector<double> dCAL2_khdT(npts);
      std::vector<double> dCAL2_khdf(npts);
      std::vector<double> dCAL2_khdB_m(npts);
      std::vector<double> d2CAL2_khdT2(npts);
      std::vector<double> d2CAL2_khdf2(npts);
      std::vector<double> d2CAL2_khdB_m2(npts);
      std::vector<double> d2CAL2_khdTdf(npts);
      std::vector<double> d2CAL2_khdTdB_m(npts);
      std::vector<double> d2CAL2_khdfdB_m(npts);
      std::vector<double> d2CAL2_khdfdT(npts);
      std::vector<double> d2CAL2_khdB_mdT(npts);
      std::vector<double> d2CAL2_khdB_mdf(npts);
      
      std::vector<double> CAL2_ke(npts);
      std::vector<double> dCAL2_kedT(npts);
      std::vector<double> dCAL2_kedf(npts);
      std::vector<double> dCAL2_kedB_m(npts);
      std::vector<double> d2CAL2_kedT2(npts);
      std::vector<double> d2CAL2_kedf2(npts);
      std::vector<double> d2CAL2_kedB_m2(npts);
      std::vector<double> d2CAL2_kedTdf(npts);
      std::vector<double> d2CAL2_kedTdB_m(npts);
      std::vector<double> d2CAL2_kedfdB_m(npts);
      std::vector<double> d2CAL2_kedfdT(npts);
      std::vector<double> d2CAL2_kedB_mdT(npts);
      std::vector<double> d2CAL2_kedB_mdf(npts);

      // for (int j = 0; j < fes.GetNE(); j++)
      for (int j = 0; j < 1; j++)
      {

         const FiniteElement &el = *fes.GetFE(j);

         IsoparametricTransformation trans;
         mesh.GetElementTransformation(j, &trans);

         const IntegrationRule *ir = NULL;
         {
            int order = trans.OrderW() + 2 * el.GetOrder();
            ir = &IntRules.Get(el.GetGeomType(), order);
         }

         // for (int i = 0; i < ir->GetNPoints(); i++)
         for (int i = 0; i < 1; i++)
         {
            const IntegrationPoint &ip = ir->IntPoint(i);

            trans.SetIntPoint(&ip);

            for (int k = 0; k < npts; ++k)
            {
               auto temperature = temperatures[k];
               auto max_flux = max_fluxes[k];

               CAL2_kh[k] = CAL2_kh_coeff.Eval(trans, ip, temperature, frequency, max_flux);
               dCAL2_khdT[k] = CAL2_kh_coeff.EvalDerivS1(trans, ip, temperature, frequency, max_flux);
               dCAL2_khdf[k] = CAL2_kh_coeff.EvalDerivS2(trans, ip, temperature, frequency, max_flux);
               dCAL2_khdB_m[k] = CAL2_kh_coeff.EvalDerivS3(trans, ip, temperature, frequency, max_flux);
               d2CAL2_khdT2[k] = CAL2_kh_coeff.Eval2ndDerivS1(trans, ip, temperature, frequency, max_flux);
               d2CAL2_khdf2[k] = CAL2_kh_coeff.Eval2ndDerivS2(trans, ip, temperature, frequency, max_flux);
               d2CAL2_khdB_m2[k] = CAL2_kh_coeff.Eval2ndDerivS3(trans, ip, temperature, frequency, max_flux);
               d2CAL2_khdTdf[k] = CAL2_kh_coeff.Eval2ndDerivS1S2(trans, ip, temperature, frequency, max_flux);
               d2CAL2_khdTdB_m[k] = CAL2_kh_coeff.Eval2ndDerivS1S3(trans, ip, temperature, frequency, max_flux);
               d2CAL2_khdfdB_m[k] = CAL2_kh_coeff.Eval2ndDerivS2S3(trans, ip, temperature, frequency, max_flux);
               d2CAL2_khdfdT[k] = CAL2_kh_coeff.Eval2ndDerivS2S1(trans, ip, temperature, frequency, max_flux);
               d2CAL2_khdB_mdT[k] = CAL2_kh_coeff.Eval2ndDerivS3S1(trans, ip, temperature, frequency, max_flux);
               d2CAL2_khdB_mdf[k] = CAL2_kh_coeff.Eval2ndDerivS3S2(trans, ip, temperature, frequency, max_flux);

               CAL2_ke[k] = CAL2_ke_coeff.Eval(trans, ip, temperature, frequency, max_flux);
               dCAL2_kedT[k] = CAL2_ke_coeff.EvalDerivS1(trans, ip, temperature, frequency, max_flux);
               dCAL2_kedf[k] = CAL2_ke_coeff.EvalDerivS2(trans, ip, temperature, frequency, max_flux);
               dCAL2_kedB_m[k] = CAL2_ke_coeff.EvalDerivS3(trans, ip, temperature, frequency, max_flux);
               d2CAL2_kedT2[k] = CAL2_ke_coeff.Eval2ndDerivS1(trans, ip, temperature, frequency, max_flux);
               d2CAL2_kedf2[k] = CAL2_ke_coeff.Eval2ndDerivS2(trans, ip, temperature, frequency, max_flux);
               d2CAL2_kedB_m2[k] = CAL2_ke_coeff.Eval2ndDerivS3(trans, ip, temperature, frequency, max_flux);
               d2CAL2_kedTdf[k] = CAL2_ke_coeff.Eval2ndDerivS1S2(trans, ip, temperature, frequency, max_flux);
               d2CAL2_kedTdB_m[k] = CAL2_ke_coeff.Eval2ndDerivS1S3(trans, ip, temperature, frequency, max_flux);
               d2CAL2_kedfdB_m[k] = CAL2_ke_coeff.Eval2ndDerivS2S3(trans, ip, temperature, frequency, max_flux);
               d2CAL2_kedfdT[k] = CAL2_ke_coeff.Eval2ndDerivS2S1(trans, ip, temperature, frequency, max_flux);
               d2CAL2_kedB_mdT[k] = CAL2_ke_coeff.Eval2ndDerivS3S1(trans, ip, temperature, frequency, max_flux);
               d2CAL2_kedB_mdf[k] = CAL2_ke_coeff.Eval2ndDerivS3S2(trans, ip, temperature, frequency, max_flux);
            }

            std::cout << "temperatures = np.array([";
            printVector(temperatures);
            std::cout << "])\n";

            std::cout << "max_fluxes = np.array([";
            printVector(max_fluxes);
            std::cout << "])\n";

            std::cout << "CAL2_kh = np.array([";
            printVector(CAL2_kh);
            std::cout << "])\n";

            // Uncomment the below if want to see derivatives in terminal output
            // std::cout << "dCAL2_khdT = np.array([";
            // printVector(dCAL2_khdT);
            // std::cout << "])\n";

            // std::cout << "dCAL2_khdf = np.array([";
            // printVector(dCAL2_khdf);
            // std::cout << "])\n";

            // std::cout << "dCAL2_khdB_m = np.array([";
            // printVector(dCAL2_khdB_m);
            // std::cout << "])\n";

            // std::cout << "d2CAL2_khdT2 = np.array([";
            // printVector(d2CAL2_khdT2);
            // std::cout << "])\n";

            // std::cout << "d2CAL2_khdf2 = np.array([";
            // printVector(d2CAL2_khdf2);
            // std::cout << "])\n";

            // std::cout << "d2CAL2_khdB_m2 = np.array([";
            // printVector(d2CAL2_khdB_m2);
            // std::cout << "])\n";

            // std::cout << "d2CAL2_khdTdf = np.array([";
            // printVector(d2CAL2_khdTdf);
            // std::cout << "])\n";

            // std::cout << "d2CAL2_khdTdB_m = np.array([";
            // printVector(d2CAL2_khdTdB_m);
            // std::cout << "])\n";

            // std::cout << "d2CAL2_khdfdB_m = np.array([";
            // printVector(d2CAL2_khdfdB_m);
            // std::cout << "])\n";

            // std::cout << "d2CAL2_khdfdT = np.array([";
            // printVector(d2CAL2_khdfdT);
            // std::cout << "])\n";

            // std::cout << "d2CAL2_khdB_mdT = np.array([";
            // printVector(d2CAL2_khdB_mdT);
            // std::cout << "])\n";

            // std::cout << "d2CAL2_khdB_mdf = np.array([";
            // printVector(d2CAL2_khdB_mdf);
            // std::cout << "])\n";

            std::cout << "CAL2_ke = np.array([";
            printVector(CAL2_ke);
            std::cout << "])\n";

            // Uncomment the below if want to see derivatives in terminal output
            // std::cout << "dCAL2_kedT = np.array([";
            // printVector(dCAL2_kedT);
            // std::cout << "])\n";

            // std::cout << "dCAL2_kedf = np.array([";
            // printVector(dCAL2_kedf);
            // std::cout << "])\n";

            // std::cout << "dCAL2_kedB_m = np.array([";
            // printVector(dCAL2_kedB_m);
            // std::cout << "])\n";

            // std::cout << "d2CAL2_kedT2 = np.array([";
            // printVector(d2CAL2_kedT2);
            // std::cout << "])\n";

            // std::cout << "d2CAL2_kedf2 = np.array([";
            // printVector(d2CAL2_kedf2);
            // std::cout << "])\n";

            // std::cout << "d2CAL2_kedB_m2 = np.array([";
            // printVector(d2CAL2_kedB_m2);
            // std::cout << "])\n";

            // std::cout << "d2CAL2_kedTdf = np.array([";
            // printVector(d2CAL2_kedTdf);
            // std::cout << "])\n";

            // std::cout << "d2CAL2_kedTdB_m = np.array([";
            // printVector(d2CAL2_kedTdB_m);
            // std::cout << "])\n";

            // std::cout << "d2CAL2_kedfdB_m = np.array([";
            // printVector(d2CAL2_kedfdB_m);
            // std::cout << "])\n";

            // std::cout << "d2CAL2_kedfdT = np.array([";
            // printVector(d2CAL2_kedfdT);
            // std::cout << "])\n";

            // std::cout << "d2CAL2_kedB_mdT = np.array([";
            // printVector(d2CAL2_kedB_mdT);
            // std::cout << "])\n";

            // std::cout << "d2CAL2_kedB_mdf = np.array([";
            // printVector(d2CAL2_kedB_mdf);
            // std::cout << "])\n";
         }
      }
   }
}

TEST_CASE("PMDemagConstraint Coefficient")
{
   using namespace mfem;
   using namespace miso;

   std::stringstream meshStr;
   meshStr << two_tet_mesh_str;
   Mesh mesh(meshStr);

   const int dim = mesh.SpaceDimension();


   /// Construct coefficient
   for (int p = 1; p <= 1; p++)
   {
      /// construct elements
      ND_FECollection fec(p, dim);
      FiniteElementSpace fes(&mesh, &fec);

      // Set the options in JSON format for the CAL2 coefficients
      const auto &pm_demag_options = R"(
      {
         "components": {
            "test": {
               "attrs": 1,
               "material": {
                  "name": "Nd2Fe14B",
                  "Demag": {
                     "T0": 293.15,
                     "alpha_B_r": -0.12,
                     "B_r_T0": 1.39,
                     "alpha_H_ci": -0.57,
                     "H_ci_T0": -1273.0,
                     "alpha_B_knee": 0.005522656,
                     "beta_B_knee": -1.4442405898,
                     "alpha_H_knee": 5.548346445,
                     "beta_H_knee": -2571.4027913402
                  }
               }
            }
         }
      })"_json;

      auto PMDemagConstraint_coeff = PMDemagConstraintCoefficient(pm_demag_options, material_library);

      int npts = 6;
      std::vector<double> C_BT(npts);
      std::vector<double> dC_BTdB(npts);
      std::vector<double> dC_BTdT(npts);
      std::vector<double> d2C_BTdB2(npts);
      std::vector<double> d2C_BTdT2(npts);
      std::vector<double> d2C_BTdBdT(npts);
      std::vector<double> d2C_BTdTdB(npts);
      // Assortment of temperatures and frequencies chosen to map to each region of the visualized constraint equation in B-T space
      // With flux_densities={0.2, 1.0, 2.0, 0.1, 0.7, 1.8}; and temperatures={90.0, 80.0, 120.0, 40.0, 50.0, 20.0};
      // expect to get C_BT={-148.197, 165.436, 725.237, 46.739, -12.416, -1788.199}
      std::vector<double> flux_densities={0.2, 1.0, 2.0, 0.1, 0.7, 1.8};  
      // std::vector<double> temperatures={90.0, 80.0, 120.0, 40.0, 50.0, 20.0};
      std::vector<double> temperatures={90.0+273.15, 80.0+273.15, 120.0+273.15, 40.0+273.15, 50.0+273.15, 20.0+273.15};
       
      for (int j = 0; j < 1; j++)
      {

         const FiniteElement &el = *fes.GetFE(j);

         IsoparametricTransformation trans;
         mesh.GetElementTransformation(j, &trans);

         const IntegrationRule *ir = NULL;
         {
            int order = trans.OrderW() + 2 * el.GetOrder();
            ir = &IntRules.Get(el.GetGeomType(), order);
         }

         for (int i = 0; i < 1; i++)
         {
            const IntegrationPoint &ip = ir->IntPoint(i);

            trans.SetIntPoint(&ip);

            for (int k = 0; k < npts; ++k)
            {
               auto flux_density = flux_densities[k];
               auto temperature = temperatures[k];

               C_BT[k] = PMDemagConstraint_coeff.Eval(trans, ip, flux_density, temperature);
               dC_BTdB[k] = PMDemagConstraint_coeff.EvalDerivS1(trans, ip, flux_density, temperature);
               dC_BTdT[k] = PMDemagConstraint_coeff.EvalDerivS2(trans, ip, flux_density, temperature);
               d2C_BTdB2[k] = PMDemagConstraint_coeff.Eval2ndDerivS1(trans, ip, flux_density, temperature);
               d2C_BTdT2[k] = PMDemagConstraint_coeff.Eval2ndDerivS2(trans, ip, flux_density, temperature);
               d2C_BTdBdT[k] = PMDemagConstraint_coeff.Eval2ndDerivS1S2(trans, ip, flux_density, temperature);
               d2C_BTdTdB[k] = PMDemagConstraint_coeff.Eval2ndDerivS2S1(trans, ip, flux_density, temperature);
            }

            std::cout << "flux densities = np.array([";
            printVector(flux_densities);
            std::cout << "])\n";

            std::cout << "temperatures = np.array([";
            printVector(temperatures);
            std::cout << "])\n";

            std::cout << "C_BT = np.array([";
            printVector(C_BT);
            std::cout << "])\n";

            // Uncomment the below if want to see derivatives in terminal output
            // std::cout << "dC_BTdB = np.array([";
            // printVector(dC_BTdB);
            // std::cout << "])\n";
            
            // std::cout << "dC_BTdT = np.array([";
            // printVector(dC_BTdT);
            // std::cout << "])\n";

            // std::cout << "d2C_BTdB2 = np.array([";
            // printVector(d2C_BTdB2);
            // std::cout << "])\n";

            // std::cout << "d2C_BTdT2 = np.array([";
            // printVector(d2C_BTdT2);
            // std::cout << "])\n";
            
            // std::cout << "d2C_BTdBdT = np.array([";
            // printVector(d2C_BTdBdT);
            // std::cout << "])\n";
            
            // std::cout << "d2C_BTdTdB = np.array([";
            // printVector(d2C_BTdTdB);
            // std::cout << "])\n";
         }
      }
   }
}

TEST_CASE("Remnant Flux Coefficient")
{
   using namespace mfem;
   using namespace miso;

   std::stringstream meshStr;
   meshStr << two_tet_mesh_str;
   Mesh mesh(meshStr);

   const int dim = mesh.SpaceDimension();


   /// Construct coefficient
   for (int p = 1; p <= 1; p++)
   {
      /// construct elements
      ND_FECollection fec(p, dim);
      FiniteElementSpace fes(&mesh, &fec);

      // Set the options in JSON format for the CAL2 coefficients
      const auto materials_material = R"(
      {
         "mu_r": 1.04,
         "B_r": 1.390,
         "alpha_B_r": -0.12,
         "T_ref": 293.15,
         "rho": 7500,
         "cv": 502.08,
         "kappa": 9,
         "max-temp": 583.15,
         "ks": 500,
         "beta": 0.0,
         "alpha": 0.0,
         "Demag": {
            "T0": 293.15,
            "alpha_B_r": -0.12,
            "B_r_T0": 1.39,
            "alpha_H_ci": -0.57,
            "H_ci_T0": -1273.0,
            "alpha_B_knee": 0.005522656,
            "beta_B_knee": 0.064272862,
            "alpha_H_knee": 5.548346445,
            "beta_H_knee": -1055.87196
         }
      }
      )"_json;

      auto B_r_coeff = RemnantFluxCoefficient(materials_material);

      int npts = 50;
      std::vector<double> temperatures(npts);
      // Temperature spans between 0 and about 200 degrees
      for (int i = 0; i < npts; ++i)
      {
         temperatures[i] = double(i)*(200/double(npts));
      }

      std::vector<double> B_r(npts);

      // for (int j = 0; j < fes.GetNE(); j++)
      for (int j = 0; j < 1; j++)
      {

         const FiniteElement &el = *fes.GetFE(j);

         IsoparametricTransformation trans;
         mesh.GetElementTransformation(j, &trans);

         const IntegrationRule *ir = NULL;
         {
            int order = trans.OrderW() + 2 * el.GetOrder();
            ir = &IntRules.Get(el.GetGeomType(), order);
         }

         // for (int i = 0; i < ir->GetNPoints(); i++)
         for (int i = 0; i < 1; i++)
         {
            const IntegrationPoint &ip = ir->IntPoint(i);

            trans.SetIntPoint(&ip);

            for (int k = 0; k < npts; ++k)
            {
               auto temperature = temperatures[k];

               B_r[k] = B_r_coeff.Eval(trans, ip, temperature);
               
            }

            std::cout << "temperatures = np.array([";
            printVector(temperatures);
            std::cout << "])\n";

            std::cout << "B_r = np.array([";
            printVector(B_r);
            std::cout << "])\n";
         }
      }
   }
}

// TEST_CASE("NonlinearReluctivityCoefficient::EvalStateDeriv",
//           "[NonlinearReluctivityCoefficient]")
// {
//    using namespace mfem;
//    using namespace miso;

//    constexpr double eps_fd = 1e-5;
//    constexpr int dim = 3;

//    std::stringstream meshStr;
//    meshStr << two_tet_mesh_str;
//    Mesh mesh(meshStr);

//    /// Costruct coefficient
//    for (int p = 1; p <= 1; p++)
//    {
//       /// construct elements
//       ND_FECollection fec(p, dim);
//       FiniteElementSpace fes(&mesh, &fec);

//       GridFunction A(&fes);
//       VectorFunctionCoefficient pert(dim, [](const Vector &x, Vector &A)
//       {
//          A(0) = -0.5*x(1);
//          A(1) = 1.79*x(0);
//          A(2) = 0.0;
//       });
//       A.ProjectCoefficient(pert);


//       auto b = material_library["hiperco50"]["B"].get<std::vector<double>>();
//       auto h = material_library["hiperco50"]["H"].get<std::vector<double>>();
//       // auto b = material_library["team13"]["B"].get<std::vector<double>>();
//       // auto h = material_library["team13"]["H"].get<std::vector<double>>();
//       miso::NonlinearReluctivityCoefficient coeff(b, h);

//       for (int j = 0; j < fes.GetNE(); j++)
//       {

//          const FiniteElement &el = *fes.GetFE(j);

//          IsoparametricTransformation trans;
//          mesh.GetElementTransformation(j, &trans);

//          const IntegrationRule *ir = NULL;
//          {
//             int order = trans.OrderW() + 2 * el.GetOrder();
//             ir = &IntRules.Get(el.GetGeomType(), order);
//          }

//          for (int i = 0; i < ir->GetNPoints(); i++)
//          {
//             const IntegrationPoint &ip = ir->IntPoint(i);

//             trans.SetIntPoint(&ip);
//             Vector b_vec;
//             A.GetCurl(trans, b_vec);

//             auto b_mag = b_vec.Norml2();

//             double dnudB = coeff.EvalStateDeriv(trans, ip, b_mag);

//             double dnudB_fd = -coeff.Eval(trans, ip, b_mag - eps_fd);
//             dnudB_fd += coeff.Eval(trans, ip, b_mag + eps_fd);
//             dnudB_fd /= (2* eps_fd);

//             // std::cout << "dnudB: " << dnudB << "\n";
//             // std::cout << "dnudB_fd: " << dnudB_fd << "\n";
//             REQUIRE(dnudB == Approx(dnudB_fd));
//          }
//       }
//    }
// }
