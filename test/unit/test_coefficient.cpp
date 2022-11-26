#include <random>
#include <vector>

#include "catch.hpp"
#include "mfem.hpp"
#include "nlohmann/json.hpp"

#include "coefficient.hpp"
#include "material_library.hpp"

#include "electromag_test_data.hpp"
#include "reluctivity_coefficient.hpp"

///TODO: Ultimately change below line to #include "conductivity_coefficient.hpp" once re-install mach and file ends up in mach/include
#include "../../src/physics/electromagnetics/conductivity_coefficient.hpp"
///TODO: Make the below absolute paths relative/shorter
#include "../../src/physics/electromagnetics/cal2_kh_coefficient.hpp"
#include "../../src/physics/electromagnetics/cal2_ke_coefficient.hpp"


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
   using namespace mach;

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

TEST_CASE("FunctionCoefficient::EvalRevDiff",
          "[FunctionCoefficient]")
{
   using namespace mfem;
   // using namespace electromag_data;
   using namespace mach;

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
   using namespace mach;

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

      mach::SteinmetzCoefficient coeff(1, 2, 4, 0.5, 0.6, A);

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
   using namespace mach;

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

      mach::SteinmetzCoefficient coeff(1, 2, 4, 0.5, 0.6, A);
      mach::SteinmetzVectorDiffCoefficient d_coeff(1, 2, 4, 0.5, 0.6, A);

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
   using namespace mach;

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
         }
      }
   }
}

TEST_CASE("ConductivityCoefficient: Models vs. Desired")
{
   using namespace mfem;
   using namespace mach;

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
                     "T_ref": 20,
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
         temperatures[i] = i;
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
         }
      }
   }
}

///TODO: Add in Steinmetz test
///TODO: Test the derivatives of the coefficients for both
TEST_CASE("CAL2 Coefficient: Models vs. Desired")
{
   using namespace mfem;
   using namespace mach;

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

      ///TODO: Determine why these options for T0, kh, ke, et al. aren't used (rather it is material library)
      const auto &CAL2_options = R"(
      {
         "components": {
            "test": {
               "attrs": 1,
               "material": {
                  "name": "hiperco50",
                  "core_loss": {
                     "model": "CAL2",
                     "T0": 20,
                     "kh_T0": [1.0, 2.0, 3.0, 4.0],
                     "ke_T0": [-1.0, -2.0, -3.0, -4.0],
                     "T1": 200,
                     "kh_T1": [10.0, 20.0, 30.0, 40.0],
                     "ke_T1": [-10.0, -20.0, -30.0, -40.0]
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
      ///TODO: Fix these vectors to produce the desired numbers
      for (int i = 0; i < npts; ++i)
      {
         temperatures[i] = 20*remainder(i+5,5);
         max_fluxes[i] = 0.2*remainder(i+10,10);
      }

      std::vector<double> CAL2_kh(npts);
      std::vector<double> CAL2_ke(npts);

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
               CAL2_ke[k] = CAL2_ke_coeff.Eval(trans, ip, temperature, frequency, max_flux);
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

            std::cout << "CAL2_ke = np.array([";
            printVector(CAL2_ke);
            std::cout << "])\n";
         }
      }
   }
}

// TEST_CASE("NonlinearReluctivityCoefficient::EvalStateDeriv",
//           "[NonlinearReluctivityCoefficient]")
// {
//    using namespace mfem;
//    using namespace mach;

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
//       mach::NonlinearReluctivityCoefficient coeff(b, h);

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
