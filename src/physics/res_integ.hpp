#ifndef MACH_RES_INTEGRATOR
#define MACH_RES_INTEGRATOR

#include "mfem.hpp"
#include "solver.hpp"

using namespace mfem;

namespace mach
{

class TestLFIntegrator : public mfem::NonlinearFormIntegrator
{
public:
   TestLFIntegrator(mfem::Coefficient &_Q)
   : Q(_Q) {}

   double GetElementEnergy(const mfem::FiniteElement &el,
                         mfem::ElementTransformation &trans,
                         const mfem::Vector &elfun) override;

private:
   mfem::Coefficient &Q;
};

class TestLFMeshSensIntegrator : public mfem::LinearFormIntegrator
{
public:
   TestLFMeshSensIntegrator(mfem::Coefficient &_Q)
   : Q(_Q) {}

   void AssembleRHSElementVect(const mfem::FiniteElement &el,
                               mfem::ElementTransformation &trans,
                               mfem::Vector &elvect) override;

private:
   mfem::Coefficient &Q;
};

/// Class that evaluates the residual part and derivatives
/// for a DomainLFIntegrator (lininteg)
class DomainResIntegrator : public mfem::NonlinearFormIntegrator
{
    Vector shape;
    Coefficient &Q;
    int oa, ob;
    GridFunction *adjoint;
public:
    /// Constructs a domain integrator with a given Coefficient
    DomainResIntegrator(Coefficient &QF, GridFunction *adj, 
                        int a = 2, int b = 0)
                        : Q(QF), oa(a), ob(b), adjoint(adj)
    { }

    /// Constructs a domain integrator with a given Coefficient
    DomainResIntegrator(Coefficient &QF, GridFunction *adj, 
                        const IntegrationRule *ir)
                        : Q(QF), oa(1), ob(1), adjoint(adj)
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
class DiffusionResIntegrator : public mfem::NonlinearFormIntegrator,
                               public mfem::LinearFormIntegrator
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
    void AssembleElementVector(const FiniteElement &elx,
                               ElementTransformation &Trx,
                               const Vector &elfunx,
                               Vector &elvect) override;


    /// Computes dR/dX, X being mesh node locations
    void AssembleRHSElementVect(const mfem::FiniteElement &el,
                                mfem::ElementTransformation &trans,
                                mfem::Vector &elvect) override;

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
                        : Q(QF), oa(a), ob(b), state(u), adjoint(adj)
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

} // namespace mach

#endif