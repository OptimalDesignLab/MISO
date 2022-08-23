#include <random>

#include "catch.hpp"
#include "mfem.hpp"
#include "adept.h"

#include "flow_residual.hpp"
#include "sbp_fe.hpp"
#include "euler_fluxes.hpp"
#include "mach_input.hpp"
#include "euler_test_data.hpp"

using namespace std;
using namespace mfem;
using namespace mach;

/// Generate a 2 Tetrahedral mesh
/// \param[in] degree - degree of polynomial used
/// \param[in] case_id - combination ID
/// \param[in] shared_nodes - number of shared nodes between these combination of elements
Mesh build2ElementTetMesh(int degree, int case_id, int &shared_nodes);

auto options = R"(
{
   "flow-param": {
      "viscous": false,
      "entropy-state": false,
      "mu": -1.0,
      "Re": 100,
      "Pr": 0.72,
      "mach": 0.5,
      "aoa": 0.0,
      "roll-axis": 0,
      "pitch-axis": 1
   },
   "space-dis": {
      "degree": 1,
      "lps-coeff": 1.0,
      "iface-coeff": 0.0,
      "basis-type": "csbp",
      "flux-fun": "Euler"
   },
   "time-dis": {
      "steady": false
   },
   "bcs": {
       "far-field": [1],
       "slip-wall": [2]
   }
      
})"_json;

TEST_CASE("FlowResidual construction and evaluation 3D - 2 tet elements", "[FlowResidual]")
{
   const int dim = 3; // templating is hard here because mesh constructors
   int num_state = dim + 2;
   adept::Stack diff_stack;
   
   int p = options["space-dis"]["degree"].get<int>();

   for (int i = 1; i <= 15; ++i)
   {  int shared_nodes = 0;
      Mesh smesh(build2ElementTetMesh(p, i, shared_nodes));
      ParMesh mesh(MPI_COMM_WORLD, smesh);
      SBPCollection fec(p, dim);
      ParFiniteElementSpace fespace(&mesh, &fec, num_state, Ordering::byVDIM);
      int ndofs = fespace.GetNDofs();
      std::map<std::string, FiniteElementState> fields;
      // construct the residual
      FlowResidual<dim,false> res(options, fespace, fields, diff_stack);
      int num_var = getSize(res);
      REQUIRE(num_var == num_state*ndofs);

      // evaluate the residual using a constant state
      Vector q(num_var);
      Vector x(num_state);
      double mach = options["flow-param"]["mach"].get<double>();
      double aoa = options["flow-param"]["aoa"].get<double>();
      for (int i = 0; i < ndofs; ++i)
      {  
         getFreeStreamQ<double, dim>(mach, aoa, 0, 1, q.GetData()+num_state*i);
      }
      auto inputs = MachInputs({{"state", q}});
      Vector res_vec2(num_var);
      ParGridFunction res_vec(&fespace);
      evaluate(res, inputs, res_vec);
      evaluate(res, inputs, res_vec2);

      // the res_vec should be zero, since we are differentiating a constant flux
      REQUIRE( res_vec2.Norml2() == Approx(0.0).margin(1e-14) );

      // check the entropy calculation; grabs the first 4 vars from q.GetData() to
      // compute the entropy, and then scales by domain size (which is 1 unit sqrd) 
      // The total_ent should be divided by 3 (since this is only 2 tet elements)
      auto total_ent = entropy<double, dim, false>(q.GetData())/3.0;
      REQUIRE( calcEntropy(res, inputs) == Approx(total_ent) );
   }      

}

TEST_CASE("FlowResidual construction and evaluation 3D - 6 Tet elements", "[FlowResidual]")
{
   const int dim = 3; // templating is hard here because mesh constructors
   int num_state = dim + 2;
   adept::Stack diff_stack;
   
   int p = options["space-dis"]["degree"].get<int>();
   // int p = 0;
   int nx = 2, ny = 2, nz = 2;
   Mesh smesh(Mesh::MakeCartesian3D(nx, ny, nz, Element::TETRAHEDRON, 1.,1.,1., Ordering::byVDIM));
   // int nv = 8, ne = 6, nb = 12, sdim = 3;
   // Mesh smesh(dim, nv, ne, nb, sdim);
   // smesh.AddVertex(Vertex(0.,0.,0.)());
   // smesh.AddVertex(Vertex(1.,0.,0.)());
   // smesh.AddVertex(Vertex(1.,1.,0.)());
   // smesh.AddVertex(Vertex(1.,0.,1.)());
   // smesh.AddVertex(Vertex(0.,0.,1.)());
   // smesh.AddVertex(Vertex(0.,1.,0.)());
   // smesh.AddVertex(Vertex(0.,1.,1.)());
   // smesh.AddVertex(Vertex(1.,1.,1.)());

   // smesh.AddElement(new Tetrahedron(0,1,2,7));
   // smesh.AddElement(new Tetrahedron(0,1,3,7));
   // smesh.AddElement(new Tetrahedron(0,3,4,7));
   // smesh.AddElement(new Tetrahedron(0,2,5,7));
   // smesh.AddElement(new Tetrahedron(0,5,6,7));
   // smesh.AddElement(new Tetrahedron(0,4,6,7));
   // smesh.AddBdrElement(new Triangle(0,1,2));
   // smesh.AddBdrElement(new Triangle(1,2,7));
   // smesh.AddBdrElement(new Triangle(0,1,3));
   // smesh.AddBdrElement(new Triangle(1,3,7));
   // smesh.AddBdrElement(new Triangle(0,3,4));
   // smesh.AddBdrElement(new Triangle(3,4,7));
   // smesh.AddBdrElement(new Triangle(0,2,5));
   // smesh.AddBdrElement(new Triangle(2,5,7));
   // smesh.AddBdrElement(new Triangle(0,5,6));
   // smesh.AddBdrElement(new Triangle(5,6,7));
   // smesh.AddBdrElement(new Triangle(0,4,6));
   // smesh.AddBdrElement(new Triangle(4,6,7));
   // smesh.FinalizeTetMesh(1,0,true);

   for (int i = 0; i < smesh.GetNBE(); ++i)
   {  
      smesh.SetBdrAttribute(i, 1);
   }

   ParMesh mesh(MPI_COMM_WORLD, smesh);
   SBPCollection fec(p, dim);
   ParFiniteElementSpace fespace(&mesh, &fec, num_state, Ordering::byVDIM);
   
   Vector n1({0.0,0.0,.25});
   Vector n2({0.0,0.0,-.25});
   // Loop through the boundary elements and compute the normals at the centers of those elements
   for (int it = 0; it < smesh.GetNBE(); ++it)
   {
   Vector normal(dim);
   ElementTransformation *Trans = smesh.GetBdrElementTransformation(it);
   Trans->SetIntPoint(&Geometries.GetCenter(Trans->GetGeometryType()));
   CalcOrtho(Trans->Jacobian(), normal);
   Vector dn1 = normal;
   Vector dn2 = normal;
   dn1 -= n1;
   dn2 -= n2;
   for  (int i=0; i < normal.Size(); ++i)
   {
      std::cout << normal(i) << " " << n1(i) << " " << dn1(i) << "\n"; 
   }
   if (dn1.Norml2() < 1e-14 || dn2.Norml2() < 1e-14)
   {
      smesh.SetBdrAttribute(it, 2);
   }
   }

   int ndofs = fespace.GetNDofs();
   std::map<std::string, FiniteElementState> fields;
   // construct the residual
   FlowResidual<dim,false> res(options, fespace, fields, diff_stack);
   int num_var = getSize(res);
   REQUIRE(num_var == num_state*ndofs);

   // evaluate the residual using a constant state
   Vector q(num_var);
   Vector x(num_state);
   double mach = options["flow-param"]["mach"].get<double>();
   double aoa = options["flow-param"]["aoa"].get<double>();
   for (int i = 0; i < ndofs; ++i)
   {  
      getFreeStreamQ<double, dim>(mach, aoa, 0, 1, q.GetData()+num_state*i);
   }
   auto inputs = MachInputs({{"state", q}});
   Vector res_vec2(num_var);
   ParGridFunction res_vec(&fespace);
   evaluate(res, inputs, res_vec);
   evaluate(res, inputs, res_vec2);

   ofstream residual("residual_values.vtk");
   residual.precision(14);
   mesh.PrintVTK(residual, 0);
   res_vec.SaveVTK(residual, "residual", 0);
   residual.close();
            
   // std::cout << "NumVar are: " << num_var << "\n";
   // int idx = 0;
   // for(int i = 0; i < res_vec2.Size(); ++i)
   // {
   //    if ((abs<double>(res_vec2(i))>1e-10))
   //    {
   //       idx = i;
   //       std::cout << "culprit node alert: " << idx/num_state << "\n";
   //    }
   //     std::cout << abs<double>(res_vec2(i)) << "\n";
   // }

   // the res_vec should be zero, since we are differentiating a constant flux
   REQUIRE( res_vec2.Norml2() == Approx(0.0).margin(1e-14) );

   // check the entropy calculation; grabs the first 4 vars from q.GetData() to
   // compute the entropy, and then scales by domain size (which is 1 unit sqrd) 
   // The total_ent should be divided by 3 (since this is only 6 tet elements)
   auto total_ent = entropy<double, dim, false>(q.GetData())/1.0;
   REQUIRE( calcEntropy(res, inputs) == Approx(total_ent) );   

}

TEST_CASE("FlowResidual calcEntropyChange - 3D tetrahedron", "[FlowResidual]")
{
   const int dim = 3; // templating is hard here because mesh constructors
   int num_state = dim + 2;
   adept::Stack diff_stack;

   // generate a 18 element periodic mesh
   int num_edge = 3;
   Mesh smesh(Mesh::MakeCartesian3D(num_edge, num_edge, num_edge, Element::TETRAHEDRON,
              1.0,1.0, 1.0, Ordering::byVDIM));
   
   // for (int i = 0; i < smesh.GetNBE(); ++i)
   // {  
   //    smesh.SetBdrAttribute(i, 1);
   // }

   Vector vx({1.0, 0.0, 0.0});
   Vector vy({0.0, 1.0, 0.0});
   Vector vz({0.0, 0.0, 1.0});
   std::vector<Vector> translations{ vx, vy, vz }; 
   auto v2v = smesh.CreatePeriodicVertexMapping(translations, 1e-12);
   Mesh periodic_smesh = Mesh::MakePeriodic(smesh, v2v);

   ParMesh mesh(MPI_COMM_WORLD, periodic_smesh);
   int p = options["space-dis"]["degree"].get<int>();
   SBPCollection fec(p, dim);
   ParFiniteElementSpace fespace(&mesh, &fec, num_state, Ordering::byVDIM);

   std::map<std::string, FiniteElementState> fields;

   // construct the residual with no dissipation and using IR flux
   options["space-dis"]["lps-coeff"] = 0.0;
   options["space-dis"]["flux-fun"] = "IR"; 
   FlowResidual<dim,false> res(options, fespace, fields, diff_stack);
   int num_var = getSize(res);

   // create a randomly perturbed conservative variable state
   static std::default_random_engine gen;
   static std::uniform_real_distribution<double> uniform_rand(0.9,1.1);
   Vector q(num_var);
   double mach = options["flow-param"]["mach"].get<double>();
   double aoa = options["flow-param"]["aoa"].get<double>();
   for (int i = 0; i < num_var/num_state; ++i)
   {
      getFreeStreamQ<double, dim>(mach, aoa, 0, 1, q.GetData()+num_state*i);
      for (int j = 0; j < num_state; ++j)
      {
         q(num_state*i + j) *= uniform_rand(gen);
      }
   }

   // evaluate the entropy change based on q; by setting dqdt to be the 
   // residual evaluated at q, we ensure the entropy change should be zero for 
   // the periodic domain and lps coeff = 0.0
   auto inputs = MachInputs({{"state", q}});
   Vector dqdt(num_var);
   evaluate(res, inputs, dqdt);
   inputs = MachInputs({
      {"state", q}, {"state_dot", dqdt}, {"time", 0.0}, {"dt", 0.0}
   });
   REQUIRE( calcEntropyChange(res, inputs) == Approx(0.0).margin(1e-14) );
}

Mesh build2ElementTetMesh(int degree, int case_id, int &shared_nodes)
{   
    int nv, ne, nb, sdim;
    int attrib = 1;

    switch (case_id)
    {
    case 1: // elements 1 and 2 combination
    {   nv = 4; ne = 2; nb = 6; sdim = 3;
        Mesh smesh1(3, nv, ne, nb, sdim);
        smesh1.AddVertex(Vertex(0.,0.,0.)());
        smesh1.AddVertex(Vertex(1.,0.,0.)());
        smesh1.AddVertex(Vertex(1.,1.,0.)());
        smesh1.AddVertex(Vertex(1.,1.,1.)());
        smesh1.AddVertex(Vertex(1.,0.,1.)());

        smesh1.AddElement(new Tetrahedron(0,1,2,3,attrib));
        smesh1.AddElement(new Tetrahedron(0,1,3,4,attrib));
        smesh1.AddBdrElement(new Triangle(0,1,2,attrib));
        smesh1.AddBdrElement(new Triangle(1,2,3,attrib));
        smesh1.AddBdrElement(new Triangle(0,2,3,attrib));
        smesh1.AddBdrElement(new Triangle(1,3,4,attrib));
        smesh1.AddBdrElement(new Triangle(0,3,4,attrib));
        smesh1.AddBdrElement(new Triangle(0,1,4,attrib));
        smesh1.FinalizeTetMesh(1,0,true); 

        if (degree == 0)
        {
            shared_nodes = 3;
        } 
        else 
        {
            shared_nodes = 7;
        }  
        return smesh1; 
        break;
    }
    
    case 2: // elements 1 and 3 combination
    {   nv = 5; ne = 2; nb = 8; sdim = 3; 
        Mesh smesh2(3, nv, ne, nb, sdim);
        smesh2.AddVertex(Vertex(0.,0.,0.)());
        smesh2.AddVertex(Vertex(1.,0.,0.)());
        smesh2.AddVertex(Vertex(1.,1.,0.)());
        smesh2.AddVertex(Vertex(1.,1.,1.)());
        smesh2.AddVertex(Vertex(0.,0.,1.)());
        smesh2.AddVertex(Vertex(1.,0.,1.)());

        smesh2.AddElement(new Tetrahedron(0,1,2,3,attrib));
        smesh2.AddElement(new Tetrahedron(0,3,4,5,attrib));
        smesh2.AddBdrElement(new Triangle(0,1,2,attrib));
        smesh2.AddBdrElement(new Triangle(1,2,3,attrib));
        smesh2.AddBdrElement(new Triangle(0,2,3,attrib));
        smesh2.AddBdrElement(new Triangle(0,1,3,attrib));
        smesh2.AddBdrElement(new Triangle(0,3,5,attrib));
        smesh2.AddBdrElement(new Triangle(0,3,4,attrib));
        smesh2.AddBdrElement(new Triangle(0,4,5,attrib));
        smesh2.AddBdrElement(new Triangle(3,4,5,attrib));
        smesh2.FinalizeTetMesh(1,0,true); 

        if (degree == 0)
        {
            shared_nodes = 2;
        } 
        else 
        {
            shared_nodes = 3;
        } 
        return smesh2;
        break;
    }
    case 3: // elements 1 and 4 combination
    {   nv = 4; ne = 2; nb = 6; sdim = 3; 
        Mesh smesh3(3, nv, ne, nb, sdim);
        smesh3.AddVertex(Vertex(0.,0.,0.)());
        smesh3.AddVertex(Vertex(1.,0.,0.)());
        smesh3.AddVertex(Vertex(1.,1.,0.)());
        smesh3.AddVertex(Vertex(1.,1.,1.)());
        smesh3.AddVertex(Vertex(0.,1.,0.)());

        smesh3.AddElement(new Tetrahedron(0,1,2,3,attrib));
        smesh3.AddElement(new Tetrahedron(0,2,3,4,attrib));
        smesh3.AddBdrElement(new Triangle(0,1,2,attrib));
        smesh3.AddBdrElement(new Triangle(1,2,3,attrib));
        smesh3.AddBdrElement(new Triangle(0,1,3,attrib));
        smesh3.AddBdrElement(new Triangle(2,3,4,attrib));
        smesh3.AddBdrElement(new Triangle(0,3,4,attrib));
        smesh3.AddBdrElement(new Triangle(0,2,4,attrib));
        smesh3.FinalizeTetMesh(1,0,true); 

        if (degree == 0)
        {
            shared_nodes = 3;
        } 
        else 
        {
            shared_nodes = 7;
        } 
        return smesh3;
        break;
    }
    case 4: // elements 1 and 5 combination
    {   nv = 5; ne = 2; nb = 8; sdim = 3;
        Mesh smesh4(3, nv, ne, nb, sdim);
        smesh4.AddVertex(Vertex(0.,0.,0.)());
        smesh4.AddVertex(Vertex(1.,0.,0.)());
        smesh4.AddVertex(Vertex(1.,1.,0.)());
        smesh4.AddVertex(Vertex(1.,1.,1.)());
        smesh4.AddVertex(Vertex(0.,1.,1.)());
        smesh4.AddVertex(Vertex(0.,1.,0.)());

        smesh4.AddElement(new Tetrahedron(0,1,2,3,attrib));
        smesh4.AddElement(new Tetrahedron(0,3,4,5,attrib));
        smesh4.AddBdrElement(new Triangle(0,1,2,attrib));
        smesh4.AddBdrElement(new Triangle(1,2,3,attrib));
        smesh4.AddBdrElement(new Triangle(0,2,3,attrib));
        smesh4.AddBdrElement(new Triangle(0,1,3,attrib));
        smesh4.AddBdrElement(new Triangle(0,3,5,attrib));
        smesh4.AddBdrElement(new Triangle(0,3,4,attrib));
        smesh4.AddBdrElement(new Triangle(0,4,5,attrib));
        smesh4.AddBdrElement(new Triangle(3,4,5,attrib));
        smesh4.FinalizeTetMesh(1,0,true); 

        if (degree == 0)
        {
            shared_nodes = 2;
        } 
        else 
        {
            shared_nodes = 3;
        } 
        return smesh4;
        break;
    }
    case 5: // elements 1 and 6 combination
    {   nv = 5; ne = 2; nb = 8; sdim = 3; 
        Mesh smesh5(3, nv, ne, nb, sdim);
        smesh5.AddVertex(Vertex(0.,0.,0.)());
        smesh5.AddVertex(Vertex(1.,0.,0.)());
        smesh5.AddVertex(Vertex(1.,1.,0.)());
        smesh5.AddVertex(Vertex(1.,1.,1.)());
        smesh5.AddVertex(Vertex(0.,1.,1.)());
        smesh5.AddVertex(Vertex(0.,0.,1.)());

        smesh5.AddElement(new Tetrahedron(0,1,2,3,attrib));
        smesh5.AddElement(new Tetrahedron(0,3,4,5,attrib));
        smesh5.AddBdrElement(new Triangle(0,1,2,attrib));
        smesh5.AddBdrElement(new Triangle(1,2,3,attrib));
        smesh5.AddBdrElement(new Triangle(0,2,3,attrib));
        smesh5.AddBdrElement(new Triangle(0,1,3,attrib));
        smesh5.AddBdrElement(new Triangle(0,3,5,attrib));
        smesh5.AddBdrElement(new Triangle(0,3,4,attrib));
        smesh5.AddBdrElement(new Triangle(0,4,5,attrib));
        smesh5.AddBdrElement(new Triangle(3,4,5,attrib));
        smesh5.FinalizeTetMesh(1,0,true); 

        if (degree == 0)
        {
            shared_nodes = 2;
        } 
        else 
        {
            shared_nodes = 3;
        } 
        return smesh5;
        break;
    }
    
    case 6: // elements 2 and 3 combination
    {   nv = 4; ne = 2; nb = 6; sdim = 3; 
        Mesh smesh(3, nv, ne, nb, sdim);
        smesh.AddVertex(Vertex(0.,0.,0.)());
        smesh.AddVertex(Vertex(1.,0.,0.)());
        smesh.AddVertex(Vertex(1.,0.,1.)());
        smesh.AddVertex(Vertex(1.,1.,1.)());
        smesh.AddVertex(Vertex(0.,0.,1.)());

        smesh.AddElement(new Tetrahedron(0,1,2,3,attrib));
        smesh.AddElement(new Tetrahedron(0,2,3,4,attrib));
        smesh.AddBdrElement(new Triangle(0,1,2,attrib));
        smesh.AddBdrElement(new Triangle(1,2,3,attrib));
        smesh.AddBdrElement(new Triangle(0,1,3,attrib));
        smesh.AddBdrElement(new Triangle(2,3,4,attrib));
        smesh.AddBdrElement(new Triangle(0,3,4,attrib));
        smesh.AddBdrElement(new Triangle(0,2,4,attrib));
        smesh.FinalizeTetMesh(1,0,true); 

        if (degree == 0)
        {
            shared_nodes = 3;
        } 
        else 
        {
            shared_nodes = 7;
        } 
        return smesh;
        break;  
    }

    case 7: // elements 2 and 4 combination 
    {   nv = 5; ne = 2; nb = 8; sdim = 3; 
        Mesh smesh(3, nv, ne, nb, sdim);
        smesh.AddVertex(Vertex(0.,0.,0.)());
        smesh.AddVertex(Vertex(1.,0.,0.)());
        smesh.AddVertex(Vertex(1.,0.,1.)());
        smesh.AddVertex(Vertex(1.,1.,1.)());
        smesh.AddVertex(Vertex(0.,1.,0.)());
        smesh.AddVertex(Vertex(1.,1.,0.)());

        smesh.AddElement(new Tetrahedron(0,1,2,3,attrib));
        smesh.AddElement(new Tetrahedron(0,3,4,5,attrib));
        smesh.AddBdrElement(new Triangle(0,1,2,attrib));
        smesh.AddBdrElement(new Triangle(1,2,3,attrib));
        smesh.AddBdrElement(new Triangle(0,2,3,attrib));
        smesh.AddBdrElement(new Triangle(0,1,3,attrib));
        smesh.AddBdrElement(new Triangle(0,3,5,attrib));
        smesh.AddBdrElement(new Triangle(0,3,4,attrib));
        smesh.AddBdrElement(new Triangle(0,4,5,attrib));
        smesh.AddBdrElement(new Triangle(3,4,5,attrib));
        smesh.FinalizeTetMesh(1,0,true); 

        if (degree == 0)
        {
            shared_nodes = 2;
        } 
        else 
        {
            shared_nodes = 3;
        } 
        return smesh;
        break;
    }

    case 8: // elements 2 and 5 combination
    {   nv = 5; ne = 2; nb = 8; sdim = 3; 
        Mesh smesh(3, nv, ne, nb, sdim);
        smesh.AddVertex(Vertex(0.,0.,0.)());
        smesh.AddVertex(Vertex(1.,0.,0.)());
        smesh.AddVertex(Vertex(1.,0.,1.)());
        smesh.AddVertex(Vertex(1.,1.,1.)());
        smesh.AddVertex(Vertex(0.,1.,0.)());
        smesh.AddVertex(Vertex(0.,1.,1.)());

        smesh.AddElement(new Tetrahedron(0,1,2,3,attrib));
        smesh.AddElement(new Tetrahedron(0,3,4,5,attrib));
        smesh.AddBdrElement(new Triangle(0,1,2,attrib));
        smesh.AddBdrElement(new Triangle(1,2,3,attrib));
        smesh.AddBdrElement(new Triangle(0,2,3,attrib));
        smesh.AddBdrElement(new Triangle(0,1,3,attrib));
        smesh.AddBdrElement(new Triangle(0,3,5,attrib));
        smesh.AddBdrElement(new Triangle(0,3,4,attrib));
        smesh.AddBdrElement(new Triangle(0,4,5,attrib));
        smesh.AddBdrElement(new Triangle(3,4,5,attrib));
        smesh.FinalizeTetMesh(1,0,true); 

        if (degree == 0)
        {
            shared_nodes = 2;
        } 
        else 
        {
            shared_nodes = 3;
        }
        return smesh;
        break; 
    }
    
    case 9: // elements 2 and 6 combination
    {   nv = 5; ne = 2; nb = 8; sdim = 3; 
        Mesh smesh(3, nv, ne, nb, sdim);
        smesh.AddVertex(Vertex(0.,0.,0.)());
        smesh.AddVertex(Vertex(1.,0.,0.)());
        smesh.AddVertex(Vertex(1.,0.,1.)());
        smesh.AddVertex(Vertex(1.,1.,1.)());
        smesh.AddVertex(Vertex(0.,1.,1.)());
        smesh.AddVertex(Vertex(0.,0.,1.)());

        smesh.AddElement(new Tetrahedron(0,1,2,3,attrib));
        smesh.AddElement(new Tetrahedron(0,3,4,5,attrib));
        smesh.AddBdrElement(new Triangle(0,1,2,attrib));
        smesh.AddBdrElement(new Triangle(1,2,3,attrib));
        smesh.AddBdrElement(new Triangle(0,2,3,attrib));
        smesh.AddBdrElement(new Triangle(0,1,3,attrib));
        smesh.AddBdrElement(new Triangle(0,3,5,attrib));
        smesh.AddBdrElement(new Triangle(0,3,4,attrib));
        smesh.AddBdrElement(new Triangle(0,4,5,attrib));
        smesh.AddBdrElement(new Triangle(3,4,5,attrib));
        smesh.FinalizeTetMesh(1,0,true); 

        if (degree == 0)
        {
            shared_nodes = 2;
        } 
        else 
        {
            shared_nodes = 3;
        }
        return smesh;
        break; 
    }

    case 10: // elements 3 and 4 combination
    {   nv = 5; ne = 2; nb = 8; sdim = 3; 
        Mesh smesh(3, nv, ne, nb, sdim);
        smesh.AddVertex(Vertex(0.,0.,0.)());
        smesh.AddVertex(Vertex(0.,0.,1.)());
        smesh.AddVertex(Vertex(1.,0.,1.)());
        smesh.AddVertex(Vertex(1.,1.,1.)());
        smesh.AddVertex(Vertex(0.,1.,0.)());
        smesh.AddVertex(Vertex(1.,1.,0.)());

        smesh.AddElement(new Tetrahedron(0,1,2,3,attrib));
        smesh.AddElement(new Tetrahedron(0,3,4,5,attrib));
        smesh.AddBdrElement(new Triangle(0,1,2,attrib));
        smesh.AddBdrElement(new Triangle(1,2,3,attrib));
        smesh.AddBdrElement(new Triangle(0,2,3,attrib));
        smesh.AddBdrElement(new Triangle(0,1,3,attrib));
        smesh.AddBdrElement(new Triangle(0,3,5,attrib));
        smesh.AddBdrElement(new Triangle(0,3,4,attrib));
        smesh.AddBdrElement(new Triangle(0,4,5,attrib));
        smesh.AddBdrElement(new Triangle(3,4,5,attrib));
        smesh.FinalizeTetMesh(1,0,true); 

        if (degree == 0)
        {
            shared_nodes = 2;
        } 
        else 
        {
            shared_nodes = 3;
        }
        return smesh;
        break; 
    }

    case 11: // elements 3 and 5 combination
    {   nv = 5; ne = 2; nb = 8; sdim = 3; 
        Mesh smesh(3, nv, ne, nb, sdim);
        smesh.AddVertex(Vertex(0.,0.,0.)());
        smesh.AddVertex(Vertex(0.,0.,1.)());
        smesh.AddVertex(Vertex(1.,0.,1.)());
        smesh.AddVertex(Vertex(1.,1.,1.)());
        smesh.AddVertex(Vertex(0.,1.,0.)());
        smesh.AddVertex(Vertex(0.,1.,1.)());

        smesh.AddElement(new Tetrahedron(0,1,2,3,attrib));
        smesh.AddElement(new Tetrahedron(0,3,4,5,attrib));
        smesh.AddBdrElement(new Triangle(0,1,2,attrib));
        smesh.AddBdrElement(new Triangle(1,2,3,attrib));
        smesh.AddBdrElement(new Triangle(0,2,3,attrib));
        smesh.AddBdrElement(new Triangle(0,1,3,attrib));
        smesh.AddBdrElement(new Triangle(0,3,5,attrib));
        smesh.AddBdrElement(new Triangle(0,3,4,attrib));
        smesh.AddBdrElement(new Triangle(0,4,5,attrib));
        smesh.AddBdrElement(new Triangle(3,4,5,attrib));
        smesh.FinalizeTetMesh(1,0,true); 

        if (degree == 0)
        {
            shared_nodes = 2;
        } 
        else 
        {
            shared_nodes = 3;
        }
        return smesh;
        break; 
    }
    
    case 12: // elements 3 and 6 combination
    {   nv = 4; ne = 2; nb = 6; sdim = 3; 
        Mesh smesh(3, nv, ne, nb, sdim);
        smesh.AddVertex(Vertex(0.,0.,0.)());
        smesh.AddVertex(Vertex(0.,0.,1.)());
        smesh.AddVertex(Vertex(1.,0.,1.)());
        smesh.AddVertex(Vertex(1.,1.,1.)());
        smesh.AddVertex(Vertex(0.,1.,1.)());

        smesh.AddElement(new Tetrahedron(0,1,2,3,attrib));
        smesh.AddElement(new Tetrahedron(0,1,3,4,attrib));
        smesh.AddBdrElement(new Triangle(0,1,2,attrib));
        smesh.AddBdrElement(new Triangle(1,2,3,attrib));
        smesh.AddBdrElement(new Triangle(0,2,3,attrib));
        smesh.AddBdrElement(new Triangle(1,3,4,attrib));
        smesh.AddBdrElement(new Triangle(0,1,4,attrib));
        smesh.AddBdrElement(new Triangle(0,3,4,attrib));
        smesh.FinalizeTetMesh(1,0,true); 

        if (degree == 0)
        {
            shared_nodes = 3;
        } 
        else 
        {
            shared_nodes = 7;
        } 
        return smesh;
        break;
    }
    
    case 13: // elements 4 and 5 combination
    {   nv = 4; ne = 2; nb = 6; sdim = 3; 
        Mesh smesh(3, nv, ne, nb, sdim);
        smesh.AddVertex(Vertex(0.,0.,0.)());
        smesh.AddVertex(Vertex(0.,1.,0.)());
        smesh.AddVertex(Vertex(1.,1.,0.)());
        smesh.AddVertex(Vertex(1.,1.,1.)());
        smesh.AddVertex(Vertex(0.,1.,1.)());

        smesh.AddElement(new Tetrahedron(0,1,2,3,attrib));
        smesh.AddElement(new Tetrahedron(0,1,3,4,attrib));
        smesh.AddBdrElement(new Triangle(0,1,2,attrib));
        smesh.AddBdrElement(new Triangle(1,2,3,attrib));
        smesh.AddBdrElement(new Triangle(0,2,3,attrib));
        smesh.AddBdrElement(new Triangle(1,3,4,attrib));
        smesh.AddBdrElement(new Triangle(0,1,4,attrib));
        smesh.AddBdrElement(new Triangle(0,3,4,attrib));
        smesh.FinalizeTetMesh(1,0,true); 

        if (degree == 0)
        {
            shared_nodes = 3;
        } 
        else 
        {
            shared_nodes = 7;
        } 
        return smesh;
        break;
    }
    
    case 14: // elements 4 and 6 combination
    {   nv = 5; ne = 2; nb = 8; sdim = 3;
        Mesh smesh(3, nv, ne, nb, sdim);
        smesh.AddVertex(Vertex(0.,0.,0.)());
        smesh.AddVertex(Vertex(0.,1.,0.)());
        smesh.AddVertex(Vertex(1.,1.,0.)());
        smesh.AddVertex(Vertex(1.,1.,1.)());
        smesh.AddVertex(Vertex(0.,0.,1.)());
        smesh.AddVertex(Vertex(0.,1.,1.)());

        smesh.AddElement(new Tetrahedron(0,1,2,3,attrib));
        smesh.AddElement(new Tetrahedron(0,3,4,5,attrib));
        smesh.AddBdrElement(new Triangle(0,1,2,attrib));
        smesh.AddBdrElement(new Triangle(1,2,3,attrib));
        smesh.AddBdrElement(new Triangle(0,2,3,attrib));
        smesh.AddBdrElement(new Triangle(0,1,3,attrib));
        smesh.AddBdrElement(new Triangle(0,3,5,attrib));
        smesh.AddBdrElement(new Triangle(0,3,4,attrib));
        smesh.AddBdrElement(new Triangle(0,4,5,attrib));
        smesh.AddBdrElement(new Triangle(3,4,5,attrib));
        smesh.FinalizeTetMesh(1,0,true); 

        if (degree == 0)
        {
            shared_nodes = 2;
        } 
        else 
        {
            shared_nodes = 3;
        }
        return smesh;
        break; 
    }

    default: // elements 5 and 6
    {   nv = 4; ne = 2; nb = 6; sdim = 3; 
        Mesh smesh(3, nv, ne, nb, sdim);
        smesh.AddVertex(Vertex(0.,0.,0.)());
        smesh.AddVertex(Vertex(0.,1.,0.)());
        smesh.AddVertex(Vertex(0.,1.,1.)());
        smesh.AddVertex(Vertex(1.,1.,1.)());
        smesh.AddVertex(Vertex(0.,0.,1.)());

        smesh.AddElement(new Tetrahedron(0,1,2,3,attrib));
        smesh.AddElement(new Tetrahedron(0,2,3,4,attrib));
        smesh.AddBdrElement(new Triangle(0,1,2,attrib));
        smesh.AddBdrElement(new Triangle(1,2,3,attrib));
        smesh.AddBdrElement(new Triangle(0,1,3,attrib));
        smesh.AddBdrElement(new Triangle(2,3,4,attrib));
        smesh.AddBdrElement(new Triangle(0,2,4,attrib));
        smesh.AddBdrElement(new Triangle(0,3,4,attrib));
        smesh.FinalizeTetMesh(1,0,true); 

        if (degree == 0)
        {
            shared_nodes = 3;
        } 
        else 
        {
            shared_nodes = 7;
        } 
        return smesh;
        break;
    }    
    }

}