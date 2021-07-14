#ifndef ELECTROMAG_TEST_DATA
#define ELECTROMAG_TEST_DATA

#include <limits>
#include <random>

#include "mfem.hpp"
#include "json.hpp"

#include "coefficient.hpp"

namespace electromag_data
{
// define the random-number generator; uniform between -1 and 1
static std::default_random_engine gen;
static std::uniform_real_distribution<double> uniform_rand(-1.0,1.0);

void randBaselineVectorPert(const mfem::Vector &x, mfem::Vector &u)
{
   const double scale = 0.5;
   for (int i = 0; i < u.Size(); ++i)
   {
      u(i) = (2.0 + scale*uniform_rand(gen));
   }
}

double randState(const mfem::Vector &x)
{
   return 2.0 * uniform_rand(gen) - 1.0;
}

void randVectorState(const mfem::Vector &x, mfem::Vector &u)
{
   // std::cout << "u size: " << u.Size() << std::endl;
   for (int i = 0; i < u.Size(); ++i)
   {
      // std::cout << i << std::endl;
      u(i) = uniform_rand(gen);
      // u(i) = 1/sqrt(3);
      // u(i) = 1.0;
   }
}

void mag_func(const mfem::Vector &x, mfem::Vector &y)
{
   y = 1.0;
}

void vectorFunc(const mfem::Vector &x, mfem::Vector &y)
{
   y.SetSize(3);
   y(0) = x(0)*x(0) - x(1);
   y(1) = x(0) * exp(x(1));
   y(2) = x(2)*x(0) - x(1);
}

void vectorFuncRevDiff(const mfem::Vector &x, const mfem::Vector &v_bar, mfem::Vector &x_bar)
{
   x_bar(0) = v_bar(0) * 2*x(0) + v_bar(1) * exp(x(1)) + v_bar(2)*x(2);
   x_bar(1) = -v_bar(0) + v_bar(1) * x(0) * exp(x(1)) - v_bar(2); 
   x_bar(2) = v_bar(2) * x(0); 
}

void vectorFunc2(const mfem::Vector &x, mfem::Vector &y)
{
   y.SetSize(3);
   y(0) = sin(x(0))*x(2)*x(2);
   y(1) = x(1) - x(0)*x(2);
   y(2) = sin(x(1))*exp(x(2));
}

void vectorFunc2RevDiff(const mfem::Vector &x, const mfem::Vector &v_bar, mfem::Vector &x_bar)
{
   x_bar(0) = cos(x(0))*x(2)*x(2)*v_bar(0) - x(2)*v_bar(1);
   x_bar(1) = v_bar(1) + cos(x(1))*exp(x(2))*v_bar(2); 
   x_bar(2) = 2*sin(x(0))*x(2)*v_bar(0) - x(0)*v_bar(1) + sin(x(1))*exp(x(2))*v_bar(2);
}

/// Simple linear coefficient for testing CurlCurlNLFIntegrator
class LinearCoefficient : public mach::StateCoefficient
{
public:
   LinearCoefficient(double val = 1.0) : value(val) {}

   double Eval(mfem::ElementTransformation &trans,
               const mfem::IntegrationPoint &ip,
               const double state) override
   {
      return value;
   }

   double EvalStateDeriv(mfem::ElementTransformation &trans,
                        const mfem::IntegrationPoint &ip,
                        const double state) override
   {
      return 0.0;
   }

private:
   double value;
};

/// Simple nonlinear coefficient for testing CurlCurlNLFIntegrator
class NonLinearCoefficient : public mach::StateCoefficient
{
public:
   NonLinearCoefficient() {};

   double Eval(mfem::ElementTransformation &trans,
               const mfem::IntegrationPoint &ip,
               const double state) override
   {
      // mfem::Vector state;
      // stateGF->GetVectorValue(trans.ElementNo, ip, state);
      // double state_mag = state.Norml2();
      // return pow(state, 2.0);
      // return state;
      return 0.5*pow(state+1, -0.5);
   }

   double EvalStateDeriv(mfem::ElementTransformation &trans,
                         const mfem::IntegrationPoint &ip,
                         const double state) override
   {
      // mfem::Vector state;
      // stateGF->GetVectorValue(trans.ElementNo, ip, state);
      // double state_mag = state.Norml2();
      // return 2.0*pow(state, 1.0);
      // return 1.0;
      return -0.25*pow(state+1, -1.5);
   }

   double EvalState2ndDeriv(mfem::ElementTransformation &trans,
                            const mfem::IntegrationPoint &ip,
                            const double state) override
   {
      // return 2.0;
      // return 0.0;
      return 0.375*pow(state+1, -2.5);
   }
};

nlohmann::json getBoxOptions(int order)
{
   nlohmann::json box_options = {
      {"silent", true},
      {"space-dis", {
         {"basis-type", "nedelec"},
         {"degree", order}
      }},
      {"steady", true},
      {"lin-solver", {
         {"type", "hypregmres"},
         {"pctype", "hypreams"},
         {"printlevel", -1},
         {"maxiter", 100},
         {"abstol", 1e-10},
         {"reltol", 1e-14}
      }},
      {"adj-solver", {
         {"type", "hypregmres"},
         {"pctype", "hypreams"},
         {"printlevel", -1},
         {"maxiter", 100},
         {"abstol", 1e-10},
         {"reltol", 1e-14}
      }},
      {"newton", {
         {"printlevel", -1},
         {"reltol", 1e-10},
         {"abstol", 0.0}
      }},
      {"components", {
         {"attr1", {
            {"material", "box1"},
            {"attr", 1},
            {"linear", true}
         }},
         {"attr2", {
            {"material", "box2"},
            {"attr", 2},
            {"linear", true}
         }}
      }},
      {"problem-opts", {
         {"fill-factor", 1.0},
         {"current-density", 1.0},
         {"current", {
            {"box1", {1}},
            {"box2", {2}}
         }},
         {"box", true}
      }},
      {"outputs", {
         {"co-energy", {""}}
      }}
   };
   return box_options;
}

std::unique_ptr<mfem::Mesh> getMesh(int nxy = 2, int nz = 2)
{
   using namespace mfem;
   // generate a simple tet mesh
   std::unique_ptr<Mesh> mesh(new Mesh(nxy, nxy, nz,
                              Element::TETRAHEDRON, true /* gen. edges */, 1.0,
                              1.0, (double)nz / (double)nxy, true));

   mesh->ReorientTetMesh();
   mesh->EnsureNodes();

   // assign attributes to top and bottom sides
   for (int i = 0; i < mesh->GetNE(); ++i)
   {
      Element *elem = mesh->GetElement(i);

      Array<int> verts;
      elem->GetVertices(verts);

      bool below = true;
      for (int i = 0; i < 4; ++i)
      {
         auto vtx = mesh->GetVertex(verts[i]);
         if (vtx[1] <= 0.5)
         {
            below = below & true;
         }
         else
         {
            below = below & false;
         }
      }
      if (below)
      {
         elem->SetAttribute(1);
      }
      else
      {
         elem->SetAttribute(2);
      }
   }
   return mesh;
}

#ifdef MFEM_USE_PUMI
nlohmann::json getWireOptions(int order)
{
   nlohmann::json wire_options = {
      {"silent", false},
      {"mesh", {
         {"file", "cut_wire.smb"},
         {"model-file", "cut_wire.egads"}
      }},
      {"space-dis", {
         {"basis-type", "nedelec"},
         {"degree", order}
      }},
      {"steady", true},
      {"lin-solver", {
         {"type", "hypregmres"},
         {"pctype", "hypreams"},
         {"printlevel", -1},
         {"maxiter", 100},
         {"abstol", 1e-10},
         {"reltol", 1e-14}
      }},
      {"adj-solver", {
         {"type", "hypregmres"},
         {"pctype", "hypreams"},
         {"printlevel", -1},
         {"maxiter", 100},
         {"abstol", 1e-10},
         {"reltol", 1e-14}
      }},
      {"newton", {
         {"printlevel", -1},
         {"reltol", 1e-10},
         {"abstol", 0.0}
      }},
      {"components", {
         {"wire", {
            {"material", "copperwire"},
            {"attrs", {1, 3, 4, 5}},
            {"linear", true}
         }},
         {"farfields", {
            {"material", "air"},
            {"attrs", {2, 6, 7, 8}},
            {"linear", true}
         }}
      }},
      {"problem-opts", {
         {"fill-factor", 1.0},
         {"current-density", 10000.0},
         {"current", {
            {"z", {1, 3, 4, 5}},
         }},
         {"box", true}
      }},
      {"outputs", {
         {"co-energy", {""}}
      }}
   };
   return wire_options;
}
#endif

} // namespace electromag_data

#endif
