#ifndef MACH_RES_INTEGRATOR
#define MACH_RES_INTEGRATOR

#include "mfem.hpp"

#include "solver.hpp"

using namespace mfem;

namespace mach
{

class DiffusionIntegratorMeshSens final : public mfem::LinearFormIntegrator
{
public:
   DiffusionIntegratorMeshSens()
      : state(nullptr), adjoint(nullptr)
   { }

   /// \brief - assemble an element's contribution to d(psi^T D u)/dX
   /// \param[in] el - the finite element that describes the mesh element
   /// \param[in] trans - the transformation between reference and physical space
   /// \param[out] mesh_coords_bar - d(psi^T D u)/dX for the element
   /// \note the LinearForm that assembles this integrator's FiniteElementSpace
   ///       MUST be the mesh's nodal finite element space
   void AssembleRHSElementVect(const mfem::FiniteElement &el,
                               mfem::ElementTransformation &trans,
                               mfem::Vector &mesh_coords_bar) override;

   void setState(const mfem::GridFunction &u)
   { state = &u; }

   void setAdjoint(const mfem::GridFunction &psi)
   { adjoint = &psi; }

private:
   /// the state to use when evaluating d(psi^T D u)/dX
   const mfem::GridFunction *state;
   /// the adjoint to use when evaluating d(psi^T D u)/dX
   const mfem::GridFunction *adjoint;
#ifndef MFEM_THREAD_SAFE
   DenseMatrix dshape, dshapedxt;
   mfem::DenseMatrix dshapedxt_bar, PointMat_bar;
   mfem::Array<int> vdofs;
   mfem::Vector elfun, psi;
#endif

};

// template <>
// inline void addSensitivityIntegrator<DiffusionIntegrator>(
//    DiffusionIntegrator &primal_integ,
//    std::unordered_map<std::string, mfem::ParGridFunction> &fields,
//    std::map<std::string, mfem::ParLinearForm> &sens,
//    std::map<std::string, mfem::ParNonlinearForm> &scalar_sens)
// {
//    auto mesh_fes = fields.at("mesh_coords").ParFESpace();
//    sens.emplace("mesh_coords", mesh_fes);
//    sens.at("mesh_coords").AddDomainIntegrator(
//       new DiffusionIntegratorMeshSens);
// };

class VectorFEWeakDivergenceIntegratorMeshSens final
   : public mfem::LinearFormIntegrator
{
public:
   VectorFEWeakDivergenceIntegratorMeshSens()
      : state(nullptr), adjoint(nullptr)
   { }

   /// \brief - assemble an element's contribution to d(psi^T W u)/dX
   /// \param[in] el - the finite element that describes the mesh element
   /// \param[in] trans - the transformation between reference and physical space
   /// \param[out] mesh_coords_bar - d(psi^T W u)/dX for the element
   /// \note the LinearForm that assembles this integrator's FiniteElementSpace
   ///       MUST be the mesh's nodal finite element space
   void AssembleRHSElementVect(const mfem::FiniteElement &el,
                               mfem::ElementTransformation &trans,
                               mfem::Vector &mesh_coords_bar) override;

   void setState(const mfem::GridFunction &u)
   { state = &u; }

   void setAdjoint(const mfem::GridFunction &psi)
   { adjoint = &psi; }

private:
   /// the state to use when evaluating d(psi^T W u)/dX
   const mfem::GridFunction *state;
   /// the adjoint to use when evaluating d(psi^T W u)/dX
   const mfem::GridFunction *adjoint;
#ifndef MFEM_THREAD_SAFE
   mfem::DenseMatrix dshape, dshapedxt, vshape, vshapedxt;
   mfem::DenseMatrix dshapedxt_bar, vshapedxt_bar, PointMat_bar;
   mfem::Array<int> vdofs;
   mfem::Vector elfun, psi;
#endif

};

// template <>
// inline void addSensitivityIntegrator<VectorFEWeakDivergenceIntegrator>(
//    VectorFEWeakDivergenceIntegrator &primal_integ,
//    std::unordered_map<std::string, mfem::ParGridFunction> &fields,
//    std::map<std::string, mfem::ParLinearForm> &sens,
//    std::map<std::string, mfem::ParNonlinearForm> &scalar_sens)
// {
//    auto mesh_fes = fields.at("mesh_coords").ParFESpace();
//    sens.emplace("mesh_coords", mesh_fes);
//    sens.at("mesh_coords").AddDomainIntegrator(
//       new VectorFEWeakDivergenceIntegratorMeshSens);
// };

class VectorFEMassIntegratorMeshSens final : public mfem::LinearFormIntegrator
{
public:
   VectorFEMassIntegratorMeshSens(double alpha = 1.0)
      : state(nullptr), adjoint(nullptr), alpha(alpha)
   { }

   /// \brief - assemble an element's contribution to d(psi^T M u)/dX
   /// \param[in] el - the finite element that describes the mesh element
   /// \param[in] trans - the transformation between reference and physical space
   /// \param[out] mesh_coords_bar - d(psi^T M u)/dX for the element
   /// \note the LinearForm that assembles this integrator's FiniteElementSpace
   ///       MUST be the mesh's nodal finite element space
   void AssembleRHSElementVect(const mfem::FiniteElement &el,
                               mfem::ElementTransformation &trans,
                               mfem::Vector &mesh_coords_bar) override;

   void setState(const mfem::GridFunction &u)
   { state = &u; }

   void setAdjoint(const mfem::GridFunction &psi)
   { adjoint = &psi; }

private:
   /// the state to use when evaluating d(psi^T M u)/dX
   const mfem::GridFunction *state;
   /// the adjoint to use when evaluating d(psi^T M u)/dX
   const mfem::GridFunction *adjoint;
   /// scaling term if the bilinear form has a negative sign in the residual
   const double alpha;
#ifndef MFEM_THREAD_SAFE
   mfem::DenseMatrix vshape, vshapedxt;
   mfem::DenseMatrix vshapedxt_bar, PointMat_bar;
   mfem::Array<int> vdofs;
   mfem::Vector elfun, psi;
#endif

};

// template <>
// inline void addSensitivityIntegrator<VectorFEMassIntegrator>(
//    VectorFEMassIntegrator &primal_integ,
//    std::unordered_map<std::string, mfem::ParGridFunction> &fields,
//    std::map<std::string, mfem::ParLinearForm> &sens,
//    std::map<std::string, mfem::ParNonlinearForm> &scalar_sens)
// {
//    auto mesh_fes = fields.at("mesh_coords").ParFESpace();
//    sens.emplace("mesh_coords", mesh_fes);

//    auto *integ = new VectorFEMassIntegratorMeshSens;
//    integ->setState(fields.at(in));
//    sens.at("mesh_coords").AddDomainIntegrator(integ);
// };

class VectorFEDomainLFIntegratorMeshSens : public mfem::LinearFormIntegrator
{
public:
   VectorFEDomainLFIntegratorMeshSens(mfem::VectorCoefficient &F, double alpha = 1.0)
      : F(F), adjoint(nullptr), alpha(alpha)
   { }

   /// \brief - assemble an element's contribution to d(psi^T f)/dX
   /// \param[in] el - the finite element that describes the mesh element
   /// \param[in] trans - the transformation between reference and physical space
   /// \param[out] mesh_coords_bar - d(psi^T f)/dX for the element
   /// \note the LinearForm that assembles this integrator's FiniteElementSpace
   ///       MUST be the mesh's nodal finite element space
   void AssembleRHSElementVect(const mfem::FiniteElement &el,
                               mfem::ElementTransformation &trans,
                               mfem::Vector &mesh_coords_bar) override;

   void setAdjoint(const mfem::GridFunction &psi)
   { adjoint = &psi; }

private:
   /// vector coefficient from linear form
   mfem::VectorCoefficient &F;
   /// the adjoint to use when evaluating d(psi^T f)/dX
   const mfem::GridFunction *adjoint;
   /// scaling term if the linear form has a negative sign in the residual
   const double alpha;
#ifndef MFEM_THREAD_SAFE
   mfem::DenseMatrix vshape, vshapedxt;
   mfem::DenseMatrix vshapedxt_bar, PointMat_bar;
   mfem::Array<int> vdofs;
   mfem::Vector psi;
#endif
};

// template <>
// inline void addSensitivityIntegrator<VectorFEDomainLFIntegrator>(
//    VectorFEDomainLFIntegrator &primal_integ,
//    std::unordered_map<std::string, mfem::ParGridFunction> &fields,
//    std::map<std::string, mfem::ParLinearForm> &sens,
//    std::map<std::string, mfem::ParNonlinearForm> &scalar_sens)
// {
//    auto mesh_fes = fields.at("mesh_coords").ParFESpace();
//    sens.emplace("mesh_coords", mesh_fes);
//    sens.at("mesh_coords").AddDomainIntegrator(
//       new VectorFEDomainLFIntegratorMeshSens);
// };

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