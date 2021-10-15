#ifndef MACH_TEST_MACH_INPUTS
#define MACH_TEST_MACH_INPUTS

#include "mfem.hpp"
#include "nlohmann/json.hpp"

#include "functional_output.hpp"
#include "solver.hpp"
#include "utils.hpp"

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

   friend void setInputs(TestMachInputIntegrator &integ,
                         const MachInputs &inputs);

private:
   double test_val;
   const mfem::GridFunction &test_field;
};

void setInputs(TestMachInputIntegrator &integ,
               const MachInputs &inputs)
{
   auto it = inputs.find("test_val");
   if (it != inputs.end())
   {
      if (it->second.isValue())
      {
         integ.test_val = it->second.getValue();
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
   void addOutputs(const std::string &fun,
                   const nlohmann::json &options)
   {
      if (fun == "testMachInput")
      {
         FunctionalOutput out(*fes, res_fields);

         out.addOutputDomainIntegrator(
            new TestMachInputIntegrator(res_fields.at("test_field")));
         outputs.emplace(fun, std::move(out));
      }
   }
   void constructForms() { res.reset(new NonlinearFormType(fes.get())); }
   int getNumState() {return 1;}
};

} // namespace mach

#endif
