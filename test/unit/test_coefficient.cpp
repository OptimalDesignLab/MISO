#include <random>

#include "catch.hpp"
#include "mfem.hpp"
#include "json.hpp"

#include "coefficient.hpp"

#include "electromag_test_data.hpp"

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

   Mesh smesh(1, 2, 2, Element::TETRAHEDRON,
              true /* gen. edges */, 1.0, 1.0, 1.0, true);

   ParMesh mesh(MPI_COMM_WORLD, smesh);

   /// Costruct coefficient
   for (int p = 1; p <= 1; p++)
   {
      /// construct elements
      ND_FECollection fec(p, dim);
      ParFiniteElementSpace fes(&mesh, &fec);


      ParGridFunction A(&fes);
      auto Aptr = &A;
      VectorFunctionCoefficient pert(dim, electromag_data::randVectorState);
      A.ProjectCoefficient(pert);

      mach::SteinmetzCoefficient coeff(1, 2, 4, 0.5, 0.6, Aptr);

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

TEST_CASE("SteinmetzVectorDiffCoefficient::Eval",
          "[SteinmetzVectorDiffCoefficient]")
{
   using namespace mfem;
   using namespace mach;

   constexpr double eps_fd = 1e-5;
   constexpr int dim = 3;

   std::stringstream meshStr;
   meshStr << two_tet_mesh_str;
   Mesh smesh(meshStr);
   ParMesh mesh(MPI_COMM_WORLD, smesh);

   /// Costruct coefficient
   for (int p = 1; p <= 1; p++)
   {
      /// construct elements
      ND_FECollection fec(p, dim);
      ParFiniteElementSpace fes(&mesh, &fec);


      ParGridFunction A(&fes);
      auto Aptr = &A;
      VectorFunctionCoefficient pert(dim, electromag_data::randVectorState);
      A.ProjectCoefficient(pert);

      mach::SteinmetzCoefficient coeff(1, 2, 4, 0.5, 0.6, Aptr);
      mach::SteinmetzVectorDiffCoefficient d_coeff(1, 2, 4, 0.5, 0.6, &A);

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


