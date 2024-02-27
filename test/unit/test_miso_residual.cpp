#include <iostream>

#include "catch.hpp"
#include "nlohmann/json.hpp"
#include "mfem.hpp"

#include "utils.hpp"
#include "miso_residual.hpp"
#include "miso_nonlinearform.hpp"
#include "flow_residual.hpp" // used in one test

class TestIntegrator : public mfem::NonlinearFormIntegrator
{
public:
   TestIntegrator()
   : time(0.0)
   { }

   // The form here is just the integral of v * cos(t) * u^2, where v is the 
   // test function, t is the time, and u is the state.  The test below sets 
   // v = 1 and u = x + y and integrates over [0,1]^2.  This should give a
   // value of 7*cos(t)/6 when `elvect` is summed.
   void AssembleElementVector(
      const mfem::FiniteElement &el,
      mfem::ElementTransformation &trans,
      const mfem::Vector &elfun,
      mfem::Vector &elvect) override
   {
      const mfem::IntegrationRule *ir = IntRule;
      if (ir == nullptr)
      {
         const int order = 2*el.GetOrder() - 2;
         ir = &mfem::IntRules.Get(el.GetGeomType(), order);
      }
      shape.SetSize(el.GetDof());
      elvect.SetSize(el.GetDof());
      elvect = 0.0;
      for (int i = 0; i < ir->GetNPoints(); i++)
      {
         const mfem::IntegrationPoint &ip = ir->IntPoint(i);
         trans.SetIntPoint(&ip);
         const double w = ip.weight * trans.Weight();
         el.CalcShape(ip, shape);
         auto u2 = pow(mfem::InnerProduct(shape, elfun), 2);
         for (int j = 0; j < el.GetDof(); ++j)
         {
            elvect(j) += shape(j) * w * cos(time) * u2;
         }
      }
   }

   // used to set the time variable for the form 
   friend void setInputs(TestIntegrator &integ, const miso::MISOInputs &inputs);

private:
   double time;
   mfem::Vector shape;
};

void setInputs(TestIntegrator &integ, const miso::MISOInputs &inputs)
{
   miso::setValueFromInputs(inputs, "time", integ.time);
}

using namespace miso;
using namespace mfem;

TEST_CASE("MISOResidual Scalar Input Test",
          "[MISOResidual]")
{
   // set up the mesh and finite-element space
   auto numx = 4;
   auto numy = 4;
   std::unique_ptr<Mesh> smesh(new Mesh(std::move(
      Mesh::MakeCartesian2D(numx, numy, Element::TRIANGLE, true, 
                            1.0, 1.0, true))));
   std::unique_ptr<ParMesh> mesh(new ParMesh(MPI_COMM_WORLD, *smesh));
   auto p = 2; // this should integrate the form exactly when u = x+y
   auto fec = H1_FECollection(p, mesh->Dimension());
   ParFiniteElementSpace fes(mesh.get(), &fec);

   // create a MISONonlinearForm and wrap it into a MISOResidual
   std::map<std::string, FiniteElementState> fields;
   MISONonlinearForm nf(fes, fields);
   nf.addDomainIntegrator(new TestIntegrator);
   MISOResidual res(std::move(nf));

   REQUIRE_NOTHROW([&](){
      MISONonlinearForm &nf_ref = getConcrete<MISONonlinearForm>(res);
   }());
   REQUIRE_THROWS([&](){
      FlowResidual<2> &flow_ref = getConcrete<FlowResidual<2>>(res);
   }());

   // These steps create a grid function for x+y
   FunctionCoefficient coeff(
      [](const mfem::Vector&xy) { return xy(0) + xy(1); });
   auto state = ParGridFunction(&fes);
   state.ProjectCoefficient(coeff);

   // set the time using a MISOInput to pi, and set state
   auto inputs = MISOInputs({
      {"time", M_PI}, {"state", state}
   });
   setInputs(res, inputs);

   // evaluate the nonlinear form, sum it, and check its value
   auto res_vec = ParGridFunction(&fes);
   evaluate(res, inputs, res_vec);
   auto integral = res_vec.Sum(); // <-- will not work on multiple processors
   REQUIRE(integral == Approx(-7.0/6.0).margin(1e-10));

   // now change the time to 1.5*pi and evaluate again
   inputs.at("time") = MISOInput(1.5*M_PI);
   setInputs(res, inputs);
   evaluate(res, inputs, res_vec);
   integral = res_vec.Sum(); // <-- will not work on multiple processors
   REQUIRE(integral == Approx(0.0).margin(1e-10));
}

