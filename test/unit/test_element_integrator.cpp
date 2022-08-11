#include <random>

#include "catch.hpp"
#include "mfem.hpp"
#include "adept.h"

#include "euler_fluxes.hpp"
#include "euler_integ.hpp"
#include "euler_test_data.hpp"
#include "mach_input.hpp"
#include "simple_integ.hpp"

using namespace std;

TEST_CASE("SIMPLE INTEGRATOR::2_ELEMENT_TETRAHEDRAL_MESH", "[SimpleIntegrator]")
{   
    using namespace mfem;
    using namespace euler_data;
    using namespace mach;

    // Generate 2 element mesh
    const int dim = 3; // templating is hard here because mesh constructors
    int num_state = 1; // just 1 state to denote values
    adept::Stack diff_stack;

    int nv = 5, ne = 2, nb = 6, sdim = 3, attrib = 1;
    Mesh smesh(dim, nv, ne, nb, sdim);
    smesh.AddVertex(Vertex(.5,.5,0.)());
    smesh.AddVertex(Vertex(.5,-.5,0.)());
    smesh.AddVertex(Vertex(-1.,0.,0.)());
    smesh.AddVertex(Vertex(0.,0.,1.)());
    smesh.AddVertex(Vertex(0.,0.,-1.)());

    smesh.AddElement(new Tetrahedron(0,1,2,3,attrib));
    smesh.AddElement(new Tetrahedron(0,1,2,4,attrib));
    smesh.AddBdrElement(new Triangle(0,1,3,attrib));
    smesh.AddBdrElement(new Triangle(0,2,3,attrib));
    smesh.AddBdrElement(new Triangle(1,2,3,attrib));
    smesh.AddBdrElement(new Triangle(0,1,4,attrib));
    smesh.AddBdrElement(new Triangle(1,2,4,attrib));
    smesh.AddBdrElement(new Triangle(0,2,4,attrib));
    smesh.FinalizeTetMesh(1,1,true); 

    for (int p = 0; p <= 1; ++p)
    {
        DYNAMIC_SECTION("... for degree p = " << p)
        {
            std::unique_ptr<FiniteElementCollection> fec(new SBPCollection(p, dim));
            std::unique_ptr<FiniteElementSpace> fes(new FiniteElementSpace(&smesh, fec.get(), num_state, Ordering::byVDIM));

            NonlinearForm res(fes.get());
            res.AddDomainIntegrator(new mach::SimpleIntegrator(num_state,1.0));

            GridFunction q(fes.get()), r(fes.get());
            q = 1.0;

            res.Mult(q,r);

            std::cout << "Norm of the residual is: " << r.Norml2() << "\n";
            for (int i = 0; i < r.Size(); ++i)
            {
                std::cout << r(i) << "\n" ;
            }

            ofstream residual("residual_values.vtk");
            residual.precision(14);
            smesh.PrintVTK(residual, 0);
            r.SaveVTK(residual, "residual", 0);
            residual.close();    

            REQUIRE(r.Norml2() == Approx(2.0).margin(1e-14));
        }
    }
}