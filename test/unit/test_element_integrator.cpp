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
using namespace mfem;

/// Generate a 2 Tetrahedral mesh
/// \param[in] degree - degree of polynomial used
/// \param[in] case_id - combination ID
/// \param[in] shared_nodes - number of shared nodes between these combination of elements
Mesh build2ElementTetMesh(int degree, int case_id, int &shared_nodes);

/// Generate a 3 Tetrahedral mesh
/// \param[in] degree - degree of polynomial used
/// \param[in] case_id - combination ID
/// \param[in] shared_nodes - number of shared nodes between these combination of elements
Mesh build3ElementTetMesh(int degree, int case_id, int &shared_nodes);

TEST_CASE("SIMPLE INTEGRATOR::2_ELEMENT_TETRAHEDRAL_MESH", "[SimpleIntegrator]")
{   
    using namespace mfem;
    using namespace euler_data;
    using namespace mach;

    // Generate 2 element mesh
    const int dim = 3; // templating is hard here because mesh constructors
    int num_state = 1; // just 1 state to denote values
    adept::Stack diff_stack;

    for (int p = 0; p <= 1; ++p)
    {
        DYNAMIC_SECTION("... for degree p = " << p)
        {   
            for (int case_id = 1; case_id <= 15; ++case_id)
            {
                DYNAMIC_SECTION("... for case_id = " << case_id)
                {   
                    int shared_nodes = 0;
                    Mesh smesh(build2ElementTetMesh(p, case_id, shared_nodes));
                
                    std::unique_ptr<FiniteElementCollection> fec(new SBPCollection(p, dim));
                    std::unique_ptr<FiniteElementSpace> fes(new FiniteElementSpace(&smesh, fec.get(), num_state, Ordering::byVDIM));

                    mfem::Vector x_chk({0.,0.,0.});
                    NonlinearForm res(fes.get());
                    res.AddDomainIntegrator(new mach::SimpleIntegrator(x_chk, num_state, 1.));

                    GridFunction q(fes.get()), r(fes.get());
                    q = 1.0;

                    res.Mult(q,r);

                    int count = 0;
                    for (int i = 0; i < r.Size(); ++i)
                    {
                        // std::cout << r(i) << "\n" ;
                        if (r(i) > 1.0)
                        {
                            ++count;
                        }
                    }
                    REQUIRE(count == shared_nodes);
                }
            }
        }
    }
}

TEST_CASE("SIMPLE INTEGRATOR::3_ELEMENT_TETRAHEDRAL_MESH", "[SimpleIntegrator]")
{   
    using namespace mfem;
    using namespace euler_data;
    using namespace mach;

    // Generate 2 element mesh
    const int dim = 3; // templating is hard here because mesh constructors
    int num_state = 1; // just 1 state to denote values
    adept::Stack diff_stack;

    for (int p = 0; p <= 1; ++p)
    {
        DYNAMIC_SECTION("... for degree p = " << p)
        {   
            for (int case_id = 1; case_id <= 19; ++case_id)
            {
                DYNAMIC_SECTION("... for case_id = " << case_id)
                {   
                    int shared_nodes = 0;
                    Mesh smesh(build3ElementTetMesh(p, case_id, shared_nodes));
                
                    std::unique_ptr<FiniteElementCollection> fec(new SBPCollection(p, dim));
                    std::unique_ptr<FiniteElementSpace> fes(new FiniteElementSpace(&smesh, fec.get(), num_state, Ordering::byVDIM));

                    mfem::Vector x_chk({0.,0.,0.});
                    NonlinearForm res(fes.get());
                    res.AddDomainIntegrator(new mach::SimpleIntegrator(x_chk, num_state, 1.));

                    GridFunction q(fes.get()), r(fes.get());
                    q = 1.0;

                    res.Mult(q,r);

                    int count = 0;
                    for (int i = 0; i < r.Size(); ++i)
                    {
                        // std::cout << r(i) << "\n" ;
                        if (r(i) > 1.0)
                        {
                            ++count;
                        }
                    }
                    REQUIRE(count == shared_nodes);
                }
            }
        }
    }
}

TEST_CASE("SIMPLE INTEGRATOR::6_ELEMENT_TETRAHEDRAL_MESH", "[SimpleIntegrator]")
{   
    using namespace mfem;
    using namespace euler_data;
    using namespace mach;

    // Generate 2 element mesh
    const int dim = 3; // templating is hard here because mesh constructors
    int num_state = 1; // just 1 state to denote values
    adept::Stack diff_stack;

    // int nx = 1, ny = 1, nz = 1;
    // Mesh smesh(Mesh::MakeCartesian3D(nx, ny, nz, Element::TETRAHEDRON, 1.,1.,1., Ordering::byNODES));
    int nv = 8, ne = 6, nb = 12, sdim = 3;
    Mesh smesh(dim, nv, ne, nb, sdim);
    smesh.AddVertex(Vertex(0.,0.,0.)());
    smesh.AddVertex(Vertex(1.,0.,0.)());
    smesh.AddVertex(Vertex(1.,1.,0.)());
    smesh.AddVertex(Vertex(1.,0.,1.)());
    smesh.AddVertex(Vertex(0.,0.,1.)());
    smesh.AddVertex(Vertex(0.,1.,0.)());
    smesh.AddVertex(Vertex(0.,1.,1.)());
    smesh.AddVertex(Vertex(1.,1.,1.)());

    smesh.AddElement(new Tetrahedron(0,1,2,7));
    smesh.AddElement(new Tetrahedron(0,1,3,7));
    smesh.AddElement(new Tetrahedron(0,3,4,7));
    smesh.AddElement(new Tetrahedron(0,2,5,7));
    smesh.AddElement(new Tetrahedron(0,5,6,7));
    smesh.AddElement(new Tetrahedron(0,4,6,7));
    smesh.AddBdrElement(new Triangle(0,1,2));
    smesh.AddBdrElement(new Triangle(1,2,7));
    smesh.AddBdrElement(new Triangle(0,1,3));
    smesh.AddBdrElement(new Triangle(1,3,7));
    smesh.AddBdrElement(new Triangle(0,3,4));
    smesh.AddBdrElement(new Triangle(3,4,7));
    smesh.AddBdrElement(new Triangle(0,2,5));
    smesh.AddBdrElement(new Triangle(2,5,7));
    smesh.AddBdrElement(new Triangle(0,5,6));
    smesh.AddBdrElement(new Triangle(5,6,7));
    smesh.AddBdrElement(new Triangle(0,4,6));
    smesh.AddBdrElement(new Triangle(4,6,7));
    smesh.FinalizeTetMesh(1,0,true);

    for (int p = 0; p <= 1; ++p)
    {
        DYNAMIC_SECTION("... for degree p = " << p)
        {   
            std::cout << "p = " << p << "\n";
            std::unique_ptr<FiniteElementCollection> fec(new SBPCollection(p, dim));
            std::unique_ptr<FiniteElementSpace> fes(new FiniteElementSpace(&smesh, fec.get(), num_state, Ordering::byVDIM));

            //mfem::Vector x_chk({1.-0.3333333333333333,0.3333333333333333,0.3333333333333333});
            mfem::Vector x_chk({0.,0.,0.});
            NonlinearForm res(fes.get());
            res.AddDomainIntegrator(new mach::SimpleIntegrator(x_chk, num_state, 1.));

            GridFunction q(fes.get()), r(fes.get());
            q = 1.0;
            res.Mult(q,r);

            int count = 0;
            for (int i = 0; i < r.Size(); ++i)
            {
                std::cout << r(i) << "\n" ;
                (r(i) > 1.0)? ++count : count = count;
            }
            int shared_nodes;
            (p==0)?shared_nodes = 8: shared_nodes = 27;
            REQUIRE(count == shared_nodes);

            ofstream residual("residual_values.vtk");
            residual.precision(14);
            smesh.PrintVTK(residual, 0);
            r.SaveVTK(residual, "residual", 0);
            residual.close();    

            //REQUIRE(r.Norml2() == Approx(2.0).margin(1e-14));
        }
    }
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

Mesh build3ElementTetMesh(int degree, int case_id, int &shared_nodes)
{
    int nv, ne = 3, nb, sdim = 3;
    int attrib = 1; 

    switch (case_id)
    {
    case 1:
    {   // elements 1, 2, and 3 combination
        nv = 6; nb = 8;
        Mesh mesh(3, nv, ne, nb, sdim);
        mesh.AddVertex(Vertex(0.,0.,0.)());
        mesh.AddVertex(Vertex(1.,0.,0.)());
        mesh.AddVertex(Vertex(1.,1.,0.)());
        mesh.AddVertex(Vertex(1.,1.,1.)());
        mesh.AddVertex(Vertex(1.,0.,1.)());
        mesh.AddVertex(Vertex(0.,0.,1.)());

        mesh.AddElement(new Tetrahedron(0,1,2,3,attrib));
        mesh.AddElement(new Tetrahedron(0,1,3,4,attrib));
        mesh.AddElement(new Tetrahedron(0,3,4,5,attrib));
        mesh.AddBdrElement(new Triangle(0,1,2,attrib));
        mesh.AddBdrElement(new Triangle(0,2,3,attrib));
        mesh.AddBdrElement(new Triangle(1,2,3,attrib));
        mesh.AddBdrElement(new Triangle(1,3,4,attrib));
        mesh.AddBdrElement(new Triangle(0,1,4,attrib));
        mesh.AddBdrElement(new Triangle(0,3,5,attrib));
        mesh.AddBdrElement(new Triangle(3,4,5,attrib));
        mesh.AddBdrElement(new Triangle(0,4,5,attrib));
        mesh.FinalizeTetMesh(1,0,true); 

        if (degree == 0)
        {
            shared_nodes = 4;
        }
        else
        {
            shared_nodes = 11;
        }
        return mesh;
        break;
    }
    case 2:
    {
        // elements 1, 2, and 4 combination
        nv = 6; nb = 8;
        Mesh mesh(3, nv, ne, nb, sdim);
        mesh.AddVertex(Vertex(0.,0.,0.)());
        mesh.AddVertex(Vertex(1.,0.,0.)());
        mesh.AddVertex(Vertex(1.,1.,0.)());
        mesh.AddVertex(Vertex(1.,1.,1.)());
        mesh.AddVertex(Vertex(1.,0.,1.)());
        mesh.AddVertex(Vertex(0.,1.,0.)());

        mesh.AddElement(new Tetrahedron(0,1,2,3,attrib));
        mesh.AddElement(new Tetrahedron(0,1,3,4,attrib));
        mesh.AddElement(new Tetrahedron(0,2,3,5,attrib));
        mesh.AddBdrElement(new Triangle(0,1,2,attrib));
        mesh.AddBdrElement(new Triangle(1,2,3,attrib));
        mesh.AddBdrElement(new Triangle(1,3,4,attrib));
        mesh.AddBdrElement(new Triangle(0,1,4,attrib));
        mesh.AddBdrElement(new Triangle(0,3,4,attrib));
        mesh.AddBdrElement(new Triangle(0,3,5,attrib));
        mesh.AddBdrElement(new Triangle(2,3,5,attrib));
        mesh.AddBdrElement(new Triangle(0,2,5,attrib));
        mesh.FinalizeTetMesh(1,0,true); 

        if (degree == 0)
        {
            shared_nodes = 4;
        }
        else
        {
            shared_nodes = 11;
        }
        return mesh;
        break;

    }
    case 3:
    {
        // elements 1, 2, and 5 combination
        nv = 7; nb = 10;
        Mesh mesh(3, nv, ne, nb, sdim);
        mesh.AddVertex(Vertex(0.,0.,0.)());
        mesh.AddVertex(Vertex(1.,0.,0.)());
        mesh.AddVertex(Vertex(1.,1.,0.)());
        mesh.AddVertex(Vertex(1.,1.,1.)());
        mesh.AddVertex(Vertex(1.,0.,1.)());
        mesh.AddVertex(Vertex(0.,1.,0.)());
        mesh.AddVertex(Vertex(0.,1.,1.)());

        mesh.AddElement(new Tetrahedron(0,1,2,3,attrib));
        mesh.AddElement(new Tetrahedron(0,1,3,4,attrib));
        mesh.AddElement(new Tetrahedron(0,3,5,6,attrib));
        mesh.AddBdrElement(new Triangle(0,1,2,attrib));
        mesh.AddBdrElement(new Triangle(0,2,3,attrib));
        mesh.AddBdrElement(new Triangle(1,2,3,attrib));
        mesh.AddBdrElement(new Triangle(1,3,4,attrib));
        mesh.AddBdrElement(new Triangle(0,1,4,attrib));
        mesh.AddBdrElement(new Triangle(0,3,4,attrib));
        mesh.AddBdrElement(new Triangle(0,3,5,attrib));
        mesh.AddBdrElement(new Triangle(3,5,6,attrib));
        mesh.AddBdrElement(new Triangle(0,5,6,attrib));
        mesh.AddBdrElement(new Triangle(0,3,6,attrib));
        mesh.FinalizeTetMesh(1,0,true); 

        if (degree == 0)
        {
            shared_nodes = 3;
        }
        else
        {
            shared_nodes = 7;
        }
        return mesh;
        break;
    }
    case 4:
    {   
        // elements 1, 2, and 6 combination
        nv = 7; nb = 10;
        Mesh mesh(3, nv, ne, nb, sdim);
        mesh.AddVertex(Vertex(0.,0.,0.)());
        mesh.AddVertex(Vertex(1.,0.,0.)());
        mesh.AddVertex(Vertex(1.,1.,0.)());
        mesh.AddVertex(Vertex(1.,1.,1.)());
        mesh.AddVertex(Vertex(1.,0.,1.)());
        mesh.AddVertex(Vertex(0.,0.,1.)());
        mesh.AddVertex(Vertex(0.,1.,1.)());

        mesh.AddElement(new Tetrahedron(0,1,2,3,attrib));
        mesh.AddElement(new Tetrahedron(0,1,3,4,attrib));
        mesh.AddElement(new Tetrahedron(0,3,5,6,attrib));
        mesh.AddBdrElement(new Triangle(0,1,2,attrib));
        mesh.AddBdrElement(new Triangle(0,2,3,attrib));
        mesh.AddBdrElement(new Triangle(1,2,3,attrib));
        mesh.AddBdrElement(new Triangle(1,3,4,attrib));
        mesh.AddBdrElement(new Triangle(0,1,4,attrib));
        mesh.AddBdrElement(new Triangle(0,3,4,attrib));
        mesh.AddBdrElement(new Triangle(0,3,5,attrib));
        mesh.AddBdrElement(new Triangle(3,5,6,attrib));
        mesh.AddBdrElement(new Triangle(0,5,6,attrib));
        mesh.AddBdrElement(new Triangle(0,3,6,attrib));
        mesh.FinalizeTetMesh(1,0,true); 

        if (degree == 0)
        {
            shared_nodes = 3;
        }
        else
        {
            shared_nodes = 7;
        }
        return mesh;
        break;
    }
    case 5:
    {
        // elements 1, 3, and 5 combination
        nv = 7; nb = 10;
        Mesh mesh(3, nv, ne, nb, sdim);
        mesh.AddVertex(Vertex(0.,0.,0.)());
        mesh.AddVertex(Vertex(1.,0.,0.)());
        mesh.AddVertex(Vertex(1.,1.,0.)());
        mesh.AddVertex(Vertex(1.,1.,1.)());
        mesh.AddVertex(Vertex(1.,0.,1.)());
        mesh.AddVertex(Vertex(0.,0.,1.)());
        mesh.AddVertex(Vertex(0.,1.,0.)());

        mesh.AddElement(new Tetrahedron(0,1,2,3,attrib));
        mesh.AddElement(new Tetrahedron(0,3,4,5,attrib));
        mesh.AddElement(new Tetrahedron(0,2,3,6,attrib));
        mesh.AddBdrElement(new Triangle(0,1,2,attrib));
        mesh.AddBdrElement(new Triangle(0,1,3,attrib));
        mesh.AddBdrElement(new Triangle(1,2,3,attrib));
        mesh.AddBdrElement(new Triangle(3,4,5,attrib));
        mesh.AddBdrElement(new Triangle(0,3,4,attrib));
        mesh.AddBdrElement(new Triangle(0,4,5,attrib));
        mesh.AddBdrElement(new Triangle(0,3,5,attrib));
        mesh.AddBdrElement(new Triangle(2,3,6,attrib));
        mesh.AddBdrElement(new Triangle(0,2,6,attrib));
        mesh.AddBdrElement(new Triangle(0,3,6,attrib));
        mesh.FinalizeTetMesh(1,0,true); 

        if (degree == 0)
        {
            shared_nodes = 3;
        }
        else
        {
            shared_nodes = 7;
        }
        return mesh;
        break;
    }
    case 6:
    {   
        // elements 1, 3, and 5 combination
        nv = 8; nb = 12;
        Mesh mesh(3, nv, ne, nb, sdim);
        mesh.AddVertex(Vertex(0.,0.,0.)());
        mesh.AddVertex(Vertex(1.,0.,0.)());
        mesh.AddVertex(Vertex(1.,1.,0.)());
        mesh.AddVertex(Vertex(1.,1.,1.)());
        mesh.AddVertex(Vertex(1.,0.,1.)());
        mesh.AddVertex(Vertex(0.,0.,1.)());
        mesh.AddVertex(Vertex(0.,1.,1.)());
        mesh.AddVertex(Vertex(0.,1.,0.)());

        mesh.AddElement(new Tetrahedron(0,1,2,3,attrib));
        mesh.AddElement(new Tetrahedron(0,3,4,5,attrib));
        mesh.AddElement(new Tetrahedron(0,3,6,7,attrib));
        mesh.AddBdrElement(new Triangle(0,1,2,attrib));
        mesh.AddBdrElement(new Triangle(0,1,3,attrib));
        mesh.AddBdrElement(new Triangle(1,2,3,attrib));
        mesh.AddBdrElement(new Triangle(0,2,3,attrib));
        mesh.AddBdrElement(new Triangle(0,3,4,attrib));
        mesh.AddBdrElement(new Triangle(0,3,5,attrib));
        mesh.AddBdrElement(new Triangle(3,4,5,attrib));
        mesh.AddBdrElement(new Triangle(0,4,5,attrib));
        mesh.AddBdrElement(new Triangle(0,3,6,attrib));
        mesh.AddBdrElement(new Triangle(0,3,7,attrib));
        mesh.AddBdrElement(new Triangle(0,6,7,attrib));
        mesh.AddBdrElement(new Triangle(3,6,7,attrib));
        mesh.FinalizeTetMesh(1,0,true);

        if (degree == 0)
        {
            shared_nodes = 2;
        }
        else
        {
            shared_nodes = 3;
        }
        return mesh;
        break;
    }
    case 7:
    {
        // elements 1, 3, and 6 combination
        nv = 7; nb = 10;
        Mesh mesh(3, nv, ne, nb, sdim);
        mesh.AddVertex(Vertex(0.,0.,0.)());
        mesh.AddVertex(Vertex(1.,0.,0.)());
        mesh.AddVertex(Vertex(1.,1.,0.)());
        mesh.AddVertex(Vertex(1.,1.,1.)());
        mesh.AddVertex(Vertex(1.,0.,1.)());
        mesh.AddVertex(Vertex(0.,0.,1.)());
        mesh.AddVertex(Vertex(0.,1.,1.)());

        mesh.AddElement(new Tetrahedron(0,1,2,3,attrib));
        mesh.AddElement(new Tetrahedron(0,3,4,5,attrib));
        mesh.AddElement(new Tetrahedron(0,3,5,6,attrib));
        mesh.AddBdrElement(new Triangle(0,1,2,attrib));
        mesh.AddBdrElement(new Triangle(0,1,3,attrib));
        mesh.AddBdrElement(new Triangle(1,2,3,attrib));
        mesh.AddBdrElement(new Triangle(0,2,3,attrib));
        mesh.AddBdrElement(new Triangle(0,3,4,attrib));
        mesh.AddBdrElement(new Triangle(0,4,5,attrib));
        mesh.AddBdrElement(new Triangle(3,4,5,attrib));
        mesh.AddBdrElement(new Triangle(3,5,6,attrib));
        mesh.AddBdrElement(new Triangle(0,5,6,attrib));
        mesh.AddBdrElement(new Triangle(0,3,6,attrib));
        mesh.FinalizeTetMesh(1,0,true); 

        if (degree == 0)
        {
            shared_nodes = 3;
        }
        else
        {
            shared_nodes = 7;
        }
        return mesh;
        break;
    }
    case 8:
    {
        // elements 1, 4, and 5 combination
        nv = 6; nb = 8;
        Mesh mesh(3, nv, ne, nb, sdim);
        mesh.AddVertex(Vertex(0.,0.,0.)());
        mesh.AddVertex(Vertex(1.,0.,0.)());
        mesh.AddVertex(Vertex(1.,1.,0.)());
        mesh.AddVertex(Vertex(1.,1.,1.)());
        mesh.AddVertex(Vertex(0.,1.,0.)());
        mesh.AddVertex(Vertex(0.,1.,1.)());

        mesh.AddElement(new Tetrahedron(0,1,2,3,attrib));
        mesh.AddElement(new Tetrahedron(0,2,3,4,attrib));
        mesh.AddElement(new Tetrahedron(0,3,4,5,attrib));
        mesh.AddBdrElement(new Triangle(0,1,2,attrib));
        mesh.AddBdrElement(new Triangle(1,2,3,attrib));
        mesh.AddBdrElement(new Triangle(0,1,3,attrib));
        mesh.AddBdrElement(new Triangle(0,2,4,attrib));
        mesh.AddBdrElement(new Triangle(2,3,4,attrib));
        mesh.AddBdrElement(new Triangle(0,4,5,attrib));
        mesh.AddBdrElement(new Triangle(0,3,5,attrib));
        mesh.AddBdrElement(new Triangle(3,4,5,attrib));
        mesh.FinalizeTetMesh(1,0,true); 

        if (degree == 0)
        {
            shared_nodes = 4;
        }
        else
        {
            shared_nodes = 11;
        }
        return mesh;
        break;
    }
    case 9:
    {
        // elements 1, 4, and 6 combination
        nv = 7; nb = 10;
        Mesh mesh(3, nv, ne, nb, sdim);
        mesh.AddVertex(Vertex(0.,0.,0.)());
        mesh.AddVertex(Vertex(1.,0.,0.)());
        mesh.AddVertex(Vertex(1.,1.,0.)());
        mesh.AddVertex(Vertex(1.,1.,1.)());
        mesh.AddVertex(Vertex(0.,1.,0.)());
        mesh.AddVertex(Vertex(0.,0.,1.)());
        mesh.AddVertex(Vertex(0.,1.,1.)());

        mesh.AddElement(new Tetrahedron(0,1,2,3,attrib));
        mesh.AddElement(new Tetrahedron(0,2,3,4,attrib));
        mesh.AddElement(new Tetrahedron(0,3,5,6,attrib));
        mesh.AddBdrElement(new Triangle(0,1,2,attrib));
        mesh.AddBdrElement(new Triangle(0,1,3,attrib));
        mesh.AddBdrElement(new Triangle(1,2,3,attrib));
        mesh.AddBdrElement(new Triangle(0,2,4,attrib));
        mesh.AddBdrElement(new Triangle(2,3,4,attrib));
        mesh.AddBdrElement(new Triangle(0,3,4,attrib));
        mesh.AddBdrElement(new Triangle(0,3,6,attrib));
        mesh.AddBdrElement(new Triangle(3,5,6,attrib));
        mesh.AddBdrElement(new Triangle(0,5,6,attrib));
        mesh.AddBdrElement(new Triangle(0,3,5,attrib));
        mesh.FinalizeTetMesh(1,0,true); 

        if (degree == 0)
        {
            shared_nodes = 3;
        }
        else
        {
            shared_nodes = 7;
        }
        return mesh;
        break;
    }
    case 10:
    {   
        // elements 1, 5, and 6 combination
        nv = 7; nb = 10;
        Mesh mesh(3, nv, ne, nb, sdim);
        mesh.AddVertex(Vertex(0.,0.,0.)());
        mesh.AddVertex(Vertex(1.,0.,0.)());
        mesh.AddVertex(Vertex(1.,1.,0.)());
        mesh.AddVertex(Vertex(1.,1.,1.)());
        mesh.AddVertex(Vertex(0.,1.,0.)());
        mesh.AddVertex(Vertex(0.,0.,1.)());
        mesh.AddVertex(Vertex(0.,1.,1.)());

        mesh.AddElement(new Tetrahedron(0,1,2,3,attrib));
        mesh.AddElement(new Tetrahedron(0,3,4,6,attrib));
        mesh.AddElement(new Tetrahedron(0,3,5,6,attrib));
        mesh.AddBdrElement(new Triangle(0,1,2,attrib));
        mesh.AddBdrElement(new Triangle(0,1,3,attrib));
        mesh.AddBdrElement(new Triangle(1,2,3,attrib));
        mesh.AddBdrElement(new Triangle(0,2,3,attrib));
        mesh.AddBdrElement(new Triangle(0,3,4,attrib));
        mesh.AddBdrElement(new Triangle(0,4,6,attrib));
        mesh.AddBdrElement(new Triangle(3,4,6,attrib));
        mesh.AddBdrElement(new Triangle(0,5,6,attrib));
        mesh.AddBdrElement(new Triangle(3,5,6,attrib));
        mesh.AddBdrElement(new Triangle(0,3,5,attrib));
        mesh.FinalizeTetMesh(1,0,true); 

        if (degree == 0)
        {
            shared_nodes = 3;
        }
        else
        {
            shared_nodes = 7;
        }
        return mesh;
        break;
    }
    case 11:
    {
        // elements 2, 3, and 4 combination
        nv = 7; nb = 10;
        Mesh mesh(3, nv, ne, nb, sdim);
        mesh.AddVertex(Vertex(0.,0.,0.)());
        mesh.AddVertex(Vertex(1.,0.,0.)());
        mesh.AddVertex(Vertex(1.,0.,1.)());
        mesh.AddVertex(Vertex(1.,1.,1.)());
        mesh.AddVertex(Vertex(0.,0.,1.)());
        mesh.AddVertex(Vertex(0.,1.,0.)());
        mesh.AddVertex(Vertex(1.,1.,0.)());

        mesh.AddElement(new Tetrahedron(0,1,2,3,attrib));
        mesh.AddElement(new Tetrahedron(0,2,3,4,attrib));
        mesh.AddElement(new Tetrahedron(0,3,5,6,attrib));
        mesh.AddBdrElement(new Triangle(0,1,2,attrib));
        mesh.AddBdrElement(new Triangle(0,1,3,attrib));
        mesh.AddBdrElement(new Triangle(1,2,3,attrib));
        mesh.AddBdrElement(new Triangle(2,3,4,attrib));
        mesh.AddBdrElement(new Triangle(0,2,4,attrib));
        mesh.AddBdrElement(new Triangle(0,3,4,attrib));
        mesh.AddBdrElement(new Triangle(0,3,5,attrib));
        mesh.AddBdrElement(new Triangle(0,5,6,attrib));
        mesh.AddBdrElement(new Triangle(3,5,6,attrib));
        mesh.AddBdrElement(new Triangle(0,3,6,attrib));
        mesh.FinalizeTetMesh(1,0,true); 

        if (degree == 0)
        {
            shared_nodes = 3;
        }
        else
        {
            shared_nodes = 7;
        }
        return mesh;
        break;
    }
    case 12:
    {
        // elements 2, 3, and 5 combination
        nv = 7; nb = 10;
        Mesh mesh(3, nv, ne, nb, sdim);
        mesh.AddVertex(Vertex(0.,0.,0.)());
        mesh.AddVertex(Vertex(1.,0.,0.)());
        mesh.AddVertex(Vertex(1.,0.,1.)());
        mesh.AddVertex(Vertex(1.,1.,1.)());
        mesh.AddVertex(Vertex(0.,0.,1.)());
        mesh.AddVertex(Vertex(0.,1.,0.)());
        mesh.AddVertex(Vertex(0.,1.,1.)());

        mesh.AddElement(new Tetrahedron(0,1,2,3,attrib));
        mesh.AddElement(new Tetrahedron(0,2,3,4,attrib));
        mesh.AddElement(new Tetrahedron(0,3,5,6,attrib));
        mesh.AddBdrElement(new Triangle(0,1,2,attrib));
        mesh.AddBdrElement(new Triangle(0,1,3,attrib));
        mesh.AddBdrElement(new Triangle(1,2,3,attrib));
        mesh.AddBdrElement(new Triangle(2,3,4,attrib));
        mesh.AddBdrElement(new Triangle(0,2,4,attrib));
        mesh.AddBdrElement(new Triangle(0,3,4,attrib));
        mesh.AddBdrElement(new Triangle(0,3,5,attrib));
        mesh.AddBdrElement(new Triangle(0,5,6,attrib));
        mesh.AddBdrElement(new Triangle(3,5,6,attrib));
        mesh.AddBdrElement(new Triangle(0,3,6,attrib));
        mesh.FinalizeTetMesh(1,0,true); 

        if (degree == 0)
        {
            shared_nodes = 3;
        }
        else
        {
            shared_nodes = 7;
        }
        return mesh;
        break;
    }
    case 13:
    {
        // elements 2, 3, and 6 combination
        nv = 6; nb = 8;
        Mesh mesh(3, nv, ne, nb, sdim);
        mesh.AddVertex(Vertex(0.,0.,0.)());
        mesh.AddVertex(Vertex(1.,0.,0.)());
        mesh.AddVertex(Vertex(1.,0.,1.)());
        mesh.AddVertex(Vertex(1.,1.,1.)());
        mesh.AddVertex(Vertex(0.,0.,1.)());
        mesh.AddVertex(Vertex(0.,1.,1.)());

        mesh.AddElement(new Tetrahedron(0,1,2,3,attrib));
        mesh.AddElement(new Tetrahedron(0,2,3,4,attrib));
        mesh.AddElement(new Tetrahedron(0,3,4,5,attrib));
        mesh.AddBdrElement(new Triangle(0,1,2,attrib));
        mesh.AddBdrElement(new Triangle(1,2,3,attrib));
        mesh.AddBdrElement(new Triangle(0,1,3,attrib));
        mesh.AddBdrElement(new Triangle(0,2,4,attrib));
        mesh.AddBdrElement(new Triangle(2,3,4,attrib));
        mesh.AddBdrElement(new Triangle(0,4,5,attrib));
        mesh.AddBdrElement(new Triangle(0,3,5,attrib));
        mesh.AddBdrElement(new Triangle(3,4,5,attrib));
        mesh.FinalizeTetMesh(1,0,true); 

        if (degree == 0)
        {
            shared_nodes = 4;
        }
        else
        {
            shared_nodes = 11;
        }
        return mesh;
        break;
    }
    case 14:
    {
        // elements 2, 4, and 5 combination
        nv = 7; nb = 10;
        Mesh mesh(3, nv, ne, nb, sdim);
        mesh.AddVertex(Vertex(0.,0.,0.)());
        mesh.AddVertex(Vertex(1.,0.,0.)());
        mesh.AddVertex(Vertex(1.,0.,1.)());
        mesh.AddVertex(Vertex(1.,1.,1.)());
        mesh.AddVertex(Vertex(0.,1.,0.)());
        mesh.AddVertex(Vertex(1.,1.,0.)());
        mesh.AddVertex(Vertex(0.,1.,1.)());

        mesh.AddElement(new Tetrahedron(0,1,2,3,attrib));
        mesh.AddElement(new Tetrahedron(0,3,4,5,attrib));
        mesh.AddElement(new Tetrahedron(0,3,4,6,attrib));
        mesh.AddBdrElement(new Triangle(0,1,2,attrib));
        mesh.AddBdrElement(new Triangle(0,1,3,attrib));
        mesh.AddBdrElement(new Triangle(1,2,3,attrib));
        mesh.AddBdrElement(new Triangle(0,2,3,attrib));
        mesh.AddBdrElement(new Triangle(0,3,5,attrib));
        mesh.AddBdrElement(new Triangle(0,4,5,attrib));
        mesh.AddBdrElement(new Triangle(3,4,5,attrib));
        mesh.AddBdrElement(new Triangle(0,3,6,attrib));
        mesh.AddBdrElement(new Triangle(0,4,6,attrib));
        mesh.AddBdrElement(new Triangle(3,4,6,attrib));
        mesh.FinalizeTetMesh(1,0,true); 

        if (degree == 0)
        {
            shared_nodes = 3;
        }
        else
        {
            shared_nodes = 7;
        }
        return mesh;
        break;
    }
    case 15:
    {
        // elements 2, 4, and 6 combination
        nv = 8; nb = 12;
        Mesh mesh(3, nv, ne, nb, sdim);
        mesh.AddVertex(Vertex(0.,0.,0.)());
        mesh.AddVertex(Vertex(1.,0.,0.)());
        mesh.AddVertex(Vertex(1.,0.,1.)());
        mesh.AddVertex(Vertex(1.,1.,1.)());
        mesh.AddVertex(Vertex(0.,1.,0.)());
        mesh.AddVertex(Vertex(1.,1.,0.)());
        mesh.AddVertex(Vertex(0.,0.,1.)());
        mesh.AddVertex(Vertex(0.,1.,1.)());

        mesh.AddElement(new Tetrahedron(0,1,2,3,attrib));
        mesh.AddElement(new Tetrahedron(0,3,4,5,attrib));
        mesh.AddElement(new Tetrahedron(0,3,6,7,attrib));
        mesh.AddBdrElement(new Triangle(0,1,2,attrib));
        mesh.AddBdrElement(new Triangle(0,1,3,attrib));
        mesh.AddBdrElement(new Triangle(1,2,3,attrib));
        mesh.AddBdrElement(new Triangle(0,2,3,attrib));
        mesh.AddBdrElement(new Triangle(0,3,4,attrib));
        mesh.AddBdrElement(new Triangle(0,3,5,attrib));
        mesh.AddBdrElement(new Triangle(3,4,5,attrib));
        mesh.AddBdrElement(new Triangle(0,4,5,attrib));
        mesh.AddBdrElement(new Triangle(0,3,6,attrib));
        mesh.AddBdrElement(new Triangle(0,3,7,attrib));
        mesh.AddBdrElement(new Triangle(0,6,7,attrib));
        mesh.AddBdrElement(new Triangle(3,6,7,attrib));
        mesh.FinalizeTetMesh(1,0,true);

        if (degree == 0)
        {
            shared_nodes = 2;
        }
        else
        {
            shared_nodes = 3;
        }
        return mesh;
        break;
    }
    case 16:
    {
        // elements 2, 5, and 6 combination
        nv = 7; nb = 10;
        Mesh mesh(3, nv, ne, nb, sdim);
        mesh.AddVertex(Vertex(0.,0.,0.)());
        mesh.AddVertex(Vertex(1.,0.,0.)());
        mesh.AddVertex(Vertex(1.,0.,1.)());
        mesh.AddVertex(Vertex(1.,1.,1.)());
        mesh.AddVertex(Vertex(0.,1.,0.)());
        mesh.AddVertex(Vertex(0.,1.,1.)());
        mesh.AddVertex(Vertex(0.,0.,1.)());

        mesh.AddElement(new Tetrahedron(0,1,2,3,attrib));
        mesh.AddElement(new Tetrahedron(0,3,4,5,attrib));
        mesh.AddElement(new Tetrahedron(0,3,5,6,attrib));
        mesh.AddBdrElement(new Triangle(0,1,2,attrib));
        mesh.AddBdrElement(new Triangle(0,1,3,attrib));
        mesh.AddBdrElement(new Triangle(1,2,3,attrib));
        mesh.AddBdrElement(new Triangle(0,2,3,attrib));
        mesh.AddBdrElement(new Triangle(0,3,4,attrib));
        mesh.AddBdrElement(new Triangle(0,4,5,attrib));
        mesh.AddBdrElement(new Triangle(3,4,5,attrib));
        mesh.AddBdrElement(new Triangle(0,3,6,attrib));
        mesh.AddBdrElement(new Triangle(0,5,6,attrib));
        mesh.AddBdrElement(new Triangle(3,5,6,attrib));
        mesh.FinalizeTetMesh(1,0,true); 

        if (degree == 0)
        {
            shared_nodes = 3;
        }
        else
        {
            shared_nodes = 7;
        }
        return mesh;
        break;
    }
    case 17:
    {
        // elements 3, 4, and 5 combination
        nv = 7; nb = 10;
        Mesh mesh(3, nv, ne, nb, sdim);
        mesh.AddVertex(Vertex(0.,0.,0.)());
        mesh.AddVertex(Vertex(0.,0.,1.)());
        mesh.AddVertex(Vertex(1.,0.,1.)());
        mesh.AddVertex(Vertex(1.,1.,1.)());
        mesh.AddVertex(Vertex(0.,1.,0.)());
        mesh.AddVertex(Vertex(1.,1.,0.)());
        mesh.AddVertex(Vertex(0.,1.,1.)());

        mesh.AddElement(new Tetrahedron(0,1,2,3,attrib));
        mesh.AddElement(new Tetrahedron(0,3,4,5,attrib));
        mesh.AddElement(new Tetrahedron(0,3,4,6,attrib));
        mesh.AddBdrElement(new Triangle(0,1,2,attrib));
        mesh.AddBdrElement(new Triangle(0,1,3,attrib));
        mesh.AddBdrElement(new Triangle(1,2,3,attrib));
        mesh.AddBdrElement(new Triangle(0,2,3,attrib));
        mesh.AddBdrElement(new Triangle(0,3,5,attrib));
        mesh.AddBdrElement(new Triangle(3,4,5,attrib));
        mesh.AddBdrElement(new Triangle(0,4,5,attrib));
        mesh.AddBdrElement(new Triangle(0,3,6,attrib));
        mesh.AddBdrElement(new Triangle(0,4,6,attrib));
        mesh.AddBdrElement(new Triangle(3,4,6,attrib));
        mesh.FinalizeTetMesh(1,0,true); 

        if (degree == 0)
        {
            shared_nodes = 3;
        }
        else
        {
            shared_nodes = 7;
        }
        return mesh;
        break;
    }
    case 18:
    {
        // elements 3, 5, and 6 combination
        nv = 6; nb = 8;
        Mesh mesh(3, nv, ne, nb, sdim);
        mesh.AddVertex(Vertex(0.,0.,0.)());
        mesh.AddVertex(Vertex(0.,0.,1.)());
        mesh.AddVertex(Vertex(1.,0.,1.)());
        mesh.AddVertex(Vertex(1.,1.,1.)());
        mesh.AddVertex(Vertex(0.,1.,0.)());
        mesh.AddVertex(Vertex(0.,1.,1.)());

        mesh.AddElement(new Tetrahedron(0,1,2,3,attrib));
        mesh.AddElement(new Tetrahedron(0,3,4,5,attrib));
        mesh.AddElement(new Tetrahedron(0,1,3,5,attrib));
        mesh.AddBdrElement(new Triangle(0,1,2,attrib));
        mesh.AddBdrElement(new Triangle(1,2,3,attrib));
        mesh.AddBdrElement(new Triangle(0,2,3,attrib));
        mesh.AddBdrElement(new Triangle(0,3,4,attrib));
        mesh.AddBdrElement(new Triangle(0,5,4,attrib));
        mesh.AddBdrElement(new Triangle(3,4,5,attrib));
        mesh.AddBdrElement(new Triangle(0,1,5,attrib));
        mesh.AddBdrElement(new Triangle(1,3,5,attrib));
        mesh.FinalizeTetMesh(1,0,true); 

        if (degree == 0)
        {
            shared_nodes = 4;
        }
        else
        {
            shared_nodes = 11;
        }
        return mesh;
        break;
    }
    case 19:
    {
        // elements 4, 5, and 6 combination
        nv = 6; nb = 8;
        Mesh mesh(3, nv, ne, nb, sdim);
        mesh.AddVertex(Vertex(0.,0.,0.)());
        mesh.AddVertex(Vertex(0.,0.,1.)());
        mesh.AddVertex(Vertex(0.,1.,1.)());
        mesh.AddVertex(Vertex(1.,1.,1.)());
        mesh.AddVertex(Vertex(0.,1.,0.)());
        mesh.AddVertex(Vertex(1.,1.,0.)());

        mesh.AddElement(new Tetrahedron(0,3,4,5,attrib));
        mesh.AddElement(new Tetrahedron(0,2,3,4,attrib));
        mesh.AddElement(new Tetrahedron(0,1,2,3,attrib));
        mesh.AddBdrElement(new Triangle(0,3,5,attrib));
        mesh.AddBdrElement(new Triangle(0,4,5,attrib));
        mesh.AddBdrElement(new Triangle(3,4,5,attrib));
        mesh.AddBdrElement(new Triangle(0,1,3,attrib));
        mesh.AddBdrElement(new Triangle(0,1,2,attrib));
        mesh.AddBdrElement(new Triangle(1,2,3,attrib));
        mesh.AddBdrElement(new Triangle(0,2,4,attrib));
        mesh.AddBdrElement(new Triangle(2,3,4,attrib));
        mesh.FinalizeTetMesh(1,0,true); 

        if (degree == 0)
        {
            shared_nodes = 4;
        }
        else
        {
            shared_nodes = 11;
        }
        return mesh;
        break;
    }
    default:
    {   
        Mesh mesh;
        shared_nodes = 0;
        return mesh;
        break;
    }
    }
}