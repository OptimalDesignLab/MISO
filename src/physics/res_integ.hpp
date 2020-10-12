#ifndef MACH_RES_INTEGRATOR
#define MACH_RES_INTEGRATOR

#include "mfem.hpp"
#include "solver.hpp"

using namespace mfem;

namespace mach
{
/// Class that evaluates the residual part and derivatives
/// for a DomainLFIntegrator (lininteg)
class DomainResIntegrator : public mfem::NonlinearFormIntegrator
{
    Vector shape;
    Coefficient &Q;
    int oa, ob;
    GridFunction *state; GridFunction *adjoint;
public:
    /// Constructs a domain integrator with a given Coefficient
    DomainResIntegrator(Coefficient &QF, GridFunction *u, GridFunction *adj, 
                        int a = 2, int b = 0)
                        : Q(QF), state(u), adjoint(adj), oa(a), ob(b)
    { }

    /// Constructs a domain integrator with a given Coefficient
    DomainResIntegrator(Coefficient &QF, GridFunction *u, GridFunction *adj, 
                        const IntegrationRule *ir)
                        : Q(QF), state(u), adjoint(adj)
    { }

    /// Computes the residual contribution
    // virtual double GetElementEnergy(const FiniteElement &elx,
    //                                    ElementTransformation &Trx,
    //                                    const Vector &elfunx);

    /// Computes dR/dX, X being mesh node locations
    virtual void AssembleElementVector(const FiniteElement &elx,
                                         ElementTransformation &Trx,
                                         const Vector &elfunx, Vector &elvect);

    /// Computes R at an integration point
    double calcFunctional(int elno, const IntegrationPoint &ip, Vector &x_q, 
                            ElementTransformation &Tr, DenseMatrix &Jac_q);

    /// Computes dR/dX at an integration point using reverse mode
    double calcFunctionalRevDiff(int elno, IntegrationPoint &ip, Vector &x_q, 
                            ElementTransformation &Tr, DenseMatrix &Jac_q,
                            Vector &x_bar, DenseMatrix &Jac_bar) {return 0.0;}

};

/// Class that evaluates the residual part and derivatives
/// for a MassIntegrator (bilininteg)
class MassResIntegrator : public mfem::NonlinearFormIntegrator
{
protected:
    Vector shape;
    Coefficient *Q;
    GridFunction *state; GridFunction *adjoint;
public:
    /// Constructs a domain integrator with a given Coefficient
    MassResIntegrator(GridFunction *u, GridFunction *adj, 
                        const IntegrationRule *ir = NULL)
                        : Q(NULL), state(u), adjoint(adj)
    { }

    /// Constructs a domain integrator with a given Coefficient
    MassResIntegrator(Coefficient &QF, GridFunction *u, GridFunction *adj, 
                        const IntegrationRule *ir = NULL)
                        : Q(&QF), state(u), adjoint(adj)
    { }

    /// Computes the residual contribution
    // virtual double GetElementEnergy(const FiniteElement &elx,
    //                                    ElementTransformation &Trx,
    //                                    const Vector &elfun);

    /// Computes dR/dX, X being mesh node locations
    virtual void AssembleElementVector(const FiniteElement &elx,
                                         ElementTransformation &Trx,
                                         const Vector &elfunx, Vector &elvect);

    /// Computes R at an integration point
    double calcFunctional(int elno, IntegrationPoint &ip, Vector &x_q, 
                            ElementTransformation &Tr, DenseMatrix &Jac_q);

    /// Computes dR/dX at an integration point using reverse mode
    double calcFunctionalRevDiff(int elno, IntegrationPoint &ip, Vector &x_q, 
                            ElementTransformation &Tr, DenseMatrix &Jac_q,
                            Vector &x_bar, DenseMatrix &Jac_bar) {return 0.0;}

};

/// Class that evaluates the residual part and derivatives
/// for a DiffusionIntegrator (bilininteg)
/// NOTE: MatrixCoefficient not implemented
class DiffusionResIntegrator : public mfem::NonlinearFormIntegrator
{
protected:
    Vector shape; DenseMatrix dshape;
    Coefficient *Q;
    GridFunction *state; GridFunction *adjoint;
public:
    /// Constructs a domain integrator with a given Coefficient
    DiffusionResIntegrator(Coefficient &QF, GridFunction *u, GridFunction *adj, 
                        const IntegrationRule *ir = NULL)
                        : Q(&QF), state(u), adjoint(adj)
    { }

    /// Computes the residual contribution
    // virtual double GetElementEnergy(const FiniteElement &elx,
    //                                    ElementTransformation &Trx,
    //                                    const Vector &elfun);

    /// Computes dR/dX, X being mesh node locations
    virtual void AssembleElementVector(const FiniteElement &elx,
                                         ElementTransformation &Trx,
                                         const Vector &elfunx, Vector &elvect);

};

/// Class that evaluates the residual part and derivatives
/// for a BoundaryNormalLFIntegrator (lininteg)
/// NOTE: Add using AddBdrFaceIntegrator
class BoundaryNormalResIntegrator : public mfem::LinearFormIntegrator
{
    Vector shape;
    VectorCoefficient &Q;
    int oa, ob;
    GridFunction *state; GridFunction *adjoint;
public:
    /// Constructs a boundary integrator with a given Coefficient
    BoundaryNormalResIntegrator(VectorCoefficient &QF, GridFunction *u, GridFunction *adj, 
                        int a = 2, int b = 0)
                        : Q(QF), state(u), adjoint(adj), oa(a), ob(b)
    { }

    /// Computes dR/dX, X being mesh node locations (DO NOT USE)
    virtual void AssembleRHSElementVect(const FiniteElement &elx,
                                         ElementTransformation &Trx,
                                         Vector &elvect) { }

    /// Computes dR/dX, X being mesh node locations
    virtual void AssembleRHSElementVect(const FiniteElement &elx,
                                         FaceElementTransformations &Trx,
                                         Vector &elvect);

};


}

#endif