#include <random>

#include "catch.hpp"
#include "mfem.hpp"

#include "electromag_test_data.hpp"
#include "miso_input.hpp"
#include "div_free_projector.hpp"

TEST_CASE("DivergenceFreeProjector::vectorJacobianProduct wrt in")
{
   using namespace mfem;
   using namespace electromag_data;

   const int dim = 3;
   double delta = 1e-5;

   int num_edge = 2;
   auto smesh = Mesh::MakeCartesian3D(num_edge, num_edge, num_edge,
                                      Element::TETRAHEDRON,
                                      1.0, 1.0, 1.0, true);

   ParMesh mesh(MPI_COMM_WORLD, smesh); 
   mesh.EnsureNodes();

   for (int p = 1; p <= 4; ++p)
   {
      DYNAMIC_SECTION( "...for degree p = " << p )
      {
         ND_FECollection fec(p, dim);
         ParFiniteElementSpace fes(&mesh, &fec);

         H1_FECollection h1_fec(p, dim);
         ParFiniteElementSpace h1_fes(&mesh, &h1_fec);

         // initialize field; here we randomly perturb a constant state
         ParGridFunction J(&fes);
         VectorFunctionCoefficient pert(3, randBaselineVectorPert);
         J.ProjectCoefficient(pert);

         auto ir_order = h1_fes.GetElementTransformation(0)->OrderW()
                           + 2 * fes.GetFE(0)->GetOrder();
         miso::DivergenceFreeProjector op(h1_fes, fes, ir_order);

         // initialize the vector that the Jacobian multiplies
         ParGridFunction p(&fes);
         p.ProjectCoefficient(pert);

         // and the vector that perturbs the input
         ParGridFunction v(&fes);
         v.ProjectCoefficient(pert);

         // evaluate the vectorJacobian product and compute its product with v
         ParGridFunction pJac(&fes); pJac = 0.0;
         op.vectorJacobianProduct(J, p, "in", pJac);
         double pJacv = pJac * v;

         // now compute the finite-difference approximation...
         ParGridFunction out(&fes);
         J.Add(delta, v);
         op.Mult(J, out);
         double pJacv_fd = p * out;
         J.Add(-2*delta, v);
         op.Mult(J, out);
         pJacv_fd -= p * out;
         pJacv_fd /= (2*delta);

         REQUIRE(pJacv == Approx(pJacv_fd));
      }
   }
}

TEST_CASE("DivergenceFreeProjector::vectorJacobianProduct wrt mesh_coords")
{
   using namespace mfem;
   using namespace electromag_data;

   const int dim = 3;
   double delta = 1e-5;

   int num_edge = 2;
   auto smesh = Mesh::MakeCartesian3D(num_edge, num_edge, num_edge,
                                      Element::TETRAHEDRON,
                                      1.0, 1.0, 1.0, true);

   ParMesh mesh(MPI_COMM_WORLD, smesh); 
   mesh.EnsureNodes();

   for (int p = 1; p <= 4; ++p)
   {
      DYNAMIC_SECTION( "...for degree p = " << p )
      {
         ND_FECollection fec(p, dim);
         ParFiniteElementSpace fes(&mesh, &fec);

         H1_FECollection h1_fec(p, dim);
         ParFiniteElementSpace h1_fes(&mesh, &h1_fec);

         // extract mesh nodes and get their finite-element space
         auto &x_nodes = dynamic_cast<ParGridFunction&>(*mesh.GetNodes());
         auto inputs = miso::MISOInputs({
            {"mesh_coords", x_nodes.GetData()}
         });
         auto &mesh_fes = *x_nodes.ParFESpace();

         // initialize field; here we randomly perturb a constant state
         ParGridFunction J(&fes);
         VectorFunctionCoefficient pert(3, randBaselineVectorPert);
         J.ProjectCoefficient(pert);

         auto ir_order = h1_fes.GetElementTransformation(0)->OrderW()
                           + 2 * fes.GetFE(0)->GetOrder();
         miso::DivergenceFreeProjector op(h1_fes, fes, ir_order);

         // initialize the vector that the Jacobian multiplies
         ParGridFunction p(&fes);
         p.ProjectCoefficient(pert);

         // and the vector that perturbs the input
         ParGridFunction v(&mesh_fes);
         v.ProjectCoefficient(pert);

         // evaluate the vectorJacobian product and compute its product with v
         ParGridFunction pJac(&mesh_fes); pJac = 0.0;
         op.vectorJacobianProduct(J, p, "mesh_coords", pJac);
         double pJacv = pJac * v;

         // now compute the finite-difference approximation...
         ParGridFunction out(&fes);
         x_nodes.Add(delta, v);
         setInputs(op, inputs);
         op.Mult(J, out);
         double pJacv_fd = p * out;
         x_nodes.Add(-2 * delta, v);
         setInputs(op, inputs);
         op.Mult(J, out);
         pJacv_fd -= p * out;
         pJacv_fd /= (2*delta);
         x_nodes.Add(delta, v); // remember to reset the mesh nodes

         std::cout << "pJacv: " << pJacv << "\n";
         std::cout << "pJacv_fd: " << pJacv_fd << "\n";
         REQUIRE(pJacv == Approx(pJacv_fd));
      }
   }
}
