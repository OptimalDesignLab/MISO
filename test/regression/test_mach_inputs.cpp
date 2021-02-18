#include <iostream>

#include "catch.hpp"
#include "json.hpp"
#include "mfem.hpp"

#include "solver.hpp"

namespace mach
{

class TestMachInputIntegrator : public mfem::NonlinearFormIntegrator
{
public:
   TestMachInputIntegrator(const mfem::GridFunction &field)
   :  test_val(0.0), test_field(field)
   { }

   double GetElementEnergy(const mfem::FiniteElement &el,
                           mfem::ElementTransformation &trans,
                           const mfem::Vector &elfun)
   {

      const mfem::IntegrationRule *ir = IntRule;
      if (ir == NULL)
      {
         const int order = 2*el.GetOrder() - 2;
         ir = &mfem::IntRules.Get(el.GetGeomType(), order);
      }

      double fun = 0.0;
      for (int i = 0; i < ir->GetNPoints(); i++)
      {
         const mfem::IntegrationPoint &ip = ir->IntPoint(i);
         trans.SetIntPoint(&ip);

         const double w = ip.weight;

         auto field_val = test_field.GetValue(trans, ip);

         fun += (field_val + test_val) * w * trans.Weight();
      }
      return fun;
   }

   friend void setInput(TestMachInputIntegrator &integ,
                        const std::string &name,
                        const MachInput &input);

private:
   double test_val;
   const mfem::GridFunction &test_field;
};

void setInput(TestMachInputIntegrator &integ,
              const std::string &name,
              const MachInput &input)
{
   if (name == "test_val")
   {
      if (input.isValue())
      {
         integ.test_val = input.getValue();
      }
      else
      {
         throw MachException("Bad input type for test_val!");
      }
   }
}

class TestMachInputSolver : public AbstractSolver
{
public:
   TestMachInputSolver(const nlohmann::json &json_options,
                       std::unique_ptr<mfem::Mesh> smesh,
                       MPI_Comm comm)
   :  AbstractSolver(json_options, move(smesh), comm)
   { }

private:
   void addOutputIntegrators(const std::string &fun,
                             const nlohmann::json &options)
   {
      if (fun == "testMachInput")
      {
         addOutputDomainIntegrator(
            fun,
            new TestMachInputIntegrator(res_fields.at("test_field")));
      }
   }
   void constructForms() { res.reset(new NonlinearFormType(fes.get())); }
   int getNumState() {return 1;}
};

} // namespace mach

auto options = R"(
{
   "silent": false,
   "print-options": false,
   "problem": "box",
   "space-dis": {
      "basis-type": "H1",
      "degree": 1
   },
   "time-dis": {
      "steady": true,
      "steady-abstol": 1e-12,
      "steady-reltol": 1e-10,
      "ode-solver": "PTC",
      "t-final": 100,
      "dt": 1e12,
      "max-iter": 10
   },
   "lin-solver": {
      "type": "hypregmres",
      "printlevel": 0,
      "maxiter": 100,
      "abstol": 1e-14,
      "reltol": 1e-14
   },
   "lin-prec": {
      "type": "hypreboomeramg",
      "printlevel": 0
   },
   "nonlin-solver": {
      "type": "newton",
      "printlevel": 3,
      "maxiter": 50,
      "reltol": 1e-10,
      "abstol": 1e-12
   },
   "external-fields": {
      "test_field": {
         "basis-type": "H1",
         "degree": 1,
         "num-states": 1
      }
   }
})"_json;

using namespace mach;
using namespace mfem;

/// Generate mesh 
/// \param[in] nxy - number of nodes in the x and y directions
/// \param[in] nz - number of nodes in the z direction
std::unique_ptr<Mesh> buildMesh(int nxy,
                                int nz);

TEST_CASE("MachInputs Scalar Input Test",
          "[MachInputs]")
{
   // construct the solver, set the initial condition, and solve
   std::unique_ptr<Mesh> mesh = buildMesh(4, 4);
   auto solver = createSolver<TestMachInputSolver>(options,
                                                   move(mesh));
   auto state = solver->getNewField();
   solver->setInitialCondition(*state, 0.0);

   auto test_field = solver->getNewField();
   solver->setInitialCondition(*test_field, 0.0);

   auto inputs = MachInputs({
      {"test_val", 2.0},
      {"test_field", test_field->GetData()},
      {"state", state->GetData()}
   });

   solver->createOutput("testMachInput");
   auto fun = solver->calcOutput("testMachInput", inputs);
   std::cout << "fun: " << fun << "\n";
   REQUIRE(fun == Approx(2.0).margin(1e-10));

   inputs.at("test_val") = 1.0;
   fun = solver->calcOutput("testMachInput", inputs);
   std::cout << "fun: " << fun << "\n";
   REQUIRE(fun == Approx(1.0).margin(1e-10));
}

TEST_CASE("MachInputs Field Input Test",
          "[MachInputs]")
{
   // construct the solver, set the initial condition, and solve
   std::unique_ptr<Mesh> mesh = buildMesh(4, 4);
   auto solver = createSolver<TestMachInputSolver>(options,
                                                   move(mesh));
   auto state = solver->getNewField();
   solver->setInitialCondition(*state, 0.0);

   auto test_field = solver->getNewField();
   solver->setInitialCondition(*test_field, 0.0);

   auto inputs = MachInputs({
      {"test_val", 0.0},
      {"test_field", test_field->GetData()},
      {"state", state->GetData()}
   });

   solver->createOutput("testMachInput");
   auto fun = solver->calcOutput("testMachInput", inputs);
   std::cout << "fun: " << fun << "\n";
   REQUIRE(fun == Approx(0.0).margin(1e-10));

   solver->setInitialCondition(*test_field, -1.0);
   fun = solver->calcOutput("testMachInput", inputs);
   std::cout << "fun: " << fun << "\n";
   REQUIRE(fun == Approx(-1.0).margin(1e-10));
}

std::unique_ptr<Mesh> buildMesh(int nxy, int nz)
{
   // generate a simple tet mesh
   std::unique_ptr<Mesh> mesh(new Mesh(nxy, nxy, nz,
                                    Element::TETRAHEDRON, true, 1.0,
                                    1.0, (double)nz / (double)nxy, true));

   mesh->ReorientTetMesh();

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
