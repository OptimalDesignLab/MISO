#ifndef MISO_TEST_MISO_INPUTS
#define MISO_TEST_MISO_INPUTS

#include "mfem.hpp"
#include "nlohmann/json.hpp"

#include "functional_output.hpp"
#include "miso_nonlinearform.hpp"
#include "pde_solver.hpp"
#include "utils.hpp"

namespace miso
{

class TestMISOInputIntegrator : public mfem::NonlinearFormIntegrator
{
public:
   TestMISOInputIntegrator(const mfem::ParGridFunction &field)
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

   friend void setInputs(TestMISOInputIntegrator &integ,
                         const MISOInputs &inputs);

private:
   double test_val;
   const mfem::ParGridFunction &test_field;
};

void setInputs(TestMISOInputIntegrator &integ,
               const MISOInputs &inputs)
{
   setValueFromInputs(inputs, "test_val", integ.test_val);
}

class TestMISOInputSolver : public PDESolver
{
public:
   TestMISOInputSolver(MPI_Comm comm,
                       const nlohmann::json &json_options,
                       std::unique_ptr<mfem::Mesh> smesh)
   :  PDESolver(comm, json_options, 1, move(smesh))
   {
      spatial_res = std::make_unique<miso::MISOResidual>(
         miso::MISONonlinearForm(fes(), fields));

      fields.emplace(
          std::piecewise_construct,
          std::forward_as_tuple("test_field"),
          std::forward_as_tuple(mesh(), fes(), "test_field"));
   }

private:
   void addOutput(const std::string &fun,
                  const nlohmann::json &options) override
   {
      if (fun == "testMISOInput")
      {
         FunctionalOutput out(fes(), fields);

         out.addOutputDomainIntegrator(
            new TestMISOInputIntegrator(fields.at("test_field").gridFunc()));
         outputs.emplace(fun, std::move(out));
      }
   }
};

} // namespace miso

#endif
