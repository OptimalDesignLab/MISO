#include <iostream>

#include "catch.hpp"
#include "nlohmann/json.hpp"
#include "mfem.hpp"

#include "utils.hpp"
#include "mach_load.hpp"
#include "mach_linearform.hpp"

// namespace
// {

class TestMachLoadIntegrator : public mfem::LinearFormIntegrator
{
public:
   TestMachLoadIntegrator()
   :  test_val(0.0)
   { }

   void AssembleRHSElementVect(const mfem::FiniteElement &el,
                               mfem::ElementTransformation &trans,
                               mfem::Vector &elvect) override
   {

      const mfem::IntegrationRule *ir = IntRule;
      if (ir == NULL)
      {
         const int order = 2*el.GetOrder() - 2;
         ir = &mfem::IntRules.Get(el.GetGeomType(), order);
      }

      elvect.SetSize(el.GetDof());
      elvect = 0.0;
      for (int i = 0; i < ir->GetNPoints(); i++)
      {
         const mfem::IntegrationPoint &ip = ir->IntPoint(i);
         trans.SetIntPoint(&ip);

         const double w = ip.weight;

         elvect += test_val * w * trans.Weight();
      }
   }

   friend void setInput(TestMachLoadIntegrator &integ,
                        const std::string &name,
                        const mach::MachInput &input);

private:
   double test_val;
};

void setInput(TestMachLoadIntegrator &integ,
              const std::string &name,
              const mach::MachInput &input)
{
   if (name == "test_val")
   {
      if (input.isValue())
      {
         integ.test_val = input.getValue();
      }
      else
      {
         throw mach::MachException("Bad input type for test_val!");
      }
   }
}

// } // anonymous namespace

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

   std::unique_ptr<Mesh> smesh = buildMesh(4, 4);
   std::unique_ptr<ParMesh> mesh(new ParMesh(MPI_COMM_WORLD, *smesh));

   auto p = 2;
   // get the finite-element space for the state
   auto fec = H1_FECollection(p, mesh->Dimension());
   ParFiniteElementSpace fes(mesh.get(), &fec);

   std::unordered_map<std::string, mfem::ParGridFunction> fields;
   MachLinearForm lf(fes, fields);
   lf.addDomainIntegrator(new TestMachLoadIntegrator);

   MachLoad ml(lf);

   auto inputs = MachInputs({
      {"test_val", 5.0}
   });

   setInputs(ml, inputs);

   HypreParVector *tv = fes.NewTrueDofVector();
   *tv = 0.0;
   addLoad(ml, *tv);

   auto norm = InnerProduct(tv, tv);
   std::cout << "norm: " << norm << "\n";
   delete tv;

   REQUIRE(norm == Approx(5.9611002604).margin(1e-10));
}

std::unique_ptr<Mesh> buildMesh(int nxy, int nz)
{
   // generate a simple tet mesh
   std::unique_ptr<Mesh> mesh(new Mesh(nxy, nxy, nz,
                              Element::TETRAHEDRON, true, 1.0,
                              1.0, (double)nz / (double)nxy, true));

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
