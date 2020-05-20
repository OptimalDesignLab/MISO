#include "res_integ.hpp"

using namespace mfem;
using namespace std;

namespace mach
{

void DomainResIntegrator::AssembleElementVector(const FiniteElement &elx,
                                       ElementTransformation &Trx,
                                       const Vector &elfunx, Vector &elvect)
{   
    /// get the proper element, transformation, and state vector
    Array<int> vdofs; Vector elfun; Vector eladj;
    int element = Trx.ElementNo;
    const FiniteElement *el = state->FESpace()->GetFE(element);
    ElementTransformation *Tr = state->FESpace()->GetElementTransformation(element);
    state->FESpace()->GetElementVDofs(element, vdofs);
    const IntegrationRule *ir = IntRule;
    if (ir == NULL)
    {
        ir = &IntRules.Get(el->GetGeomType(), oa * el->GetOrder() + ob);
    }
    state->GetSubVector(vdofs, elfun); //don't need this one
    adjoint->GetSubVector(vdofs, eladj);

    const int dof = elx.GetDof();
    const int dofu = el->GetDof();
    const int dim = el->GetDim();
    elvect.SetSize(dof*dim);
    elvect = 0.0;
    shape.SetSize(dofu);

    // cast the ElementTransformation
    IsoparametricTransformation &isotrans =
    dynamic_cast<IsoparametricTransformation&>(*Tr);

    Vector x_q(dim);
    Vector x_bar(dim);
    DenseMatrix Jac_q(dim, dim);
    DenseMatrix Jac_bar(dim, dim);
    DenseMatrix PointMat_bar(dim, dof);

    
    // loop through nodes
    for (int i = 0; i < ir->GetNPoints(); ++i)
    {
        const IntegrationPoint &ip = ir->IntPoint(i);
        Tr->SetIntPoint(&ip);
        Tr->Transform(ip, x_q);

        PointMat_bar = 0.0;

        /// NOTE: Q may or may not have sensitivity to x. Need to tailor to
        /// different coefficients 
        el->CalcShape(ip, shape);
        double deriv = ip.weight*Q.Eval(*Tr, ip)*(eladj*shape); //dR/dWeight
        isotrans.WeightRevDiff(PointMat_bar); //dWeight/dX        
        PointMat_bar.Set(deriv, PointMat_bar);

        /// Implement Q sensitivity
        double Q_bar = ip.weight*Tr->Weight()*(eladj*shape); //dR/dQ
        Q.EvalRevDiff(Q_bar, *Tr, ip, PointMat_bar);

        for (int j = 0; j < dof ; ++j)
        {
            for (int d = 0; d < dim; ++d)
            {
                elvect(d* dof + j) += PointMat_bar (d,j);
            }
        }
    }
}

void MassResIntegrator::AssembleElementVector(const FiniteElement &elx,
                                       ElementTransformation &Trx,
                                       const Vector &elfunx, Vector &elvect)
{
    /// get the proper element, transformation, and state vector
    Array<int> vdofs; Vector elfun; Vector eladj; 
    int element = Trx.ElementNo;
    const FiniteElement *el = state->FESpace()->GetFE(element);
    ElementTransformation *Tr = state->FESpace()->GetElementTransformation(element);
    state->FESpace()->GetElementVDofs(element, vdofs);
    int order = 2*el->GetOrder() + Tr->OrderW();
    const IntegrationRule *ir = IntRule;
    if (ir == NULL)
    {
        ir = &IntRules.Get(el->GetGeomType(), order);
    }
    state->GetSubVector(vdofs, elfun);
    adjoint->GetSubVector(vdofs, eladj);

    const int dof = elx.GetDof();
    const int dofu = el->GetDof();
    const int dim = el->GetDim();
    elvect.SetSize(dof*dim);
    elvect = 0.0;
    shape.SetSize(dofu);

    // cast the ElementTransformation
    IsoparametricTransformation &isotrans =
    dynamic_cast<IsoparametricTransformation&>(*Tr);

    DenseMatrix elmat(dofu);
    DenseMatrix PointMat_bar(dim, dof);
    
    // loop through nodes
    for (int i = 0; i < ir->GetNPoints(); ++i)
    {
        const IntegrationPoint &ip = ir->IntPoint(i);
        Tr->SetIntPoint(&ip);

        PointMat_bar = 0.0;

        /// NOTE: Q may or may not have sensitivity to x. Need to tailor to
        /// different coefficients 
        double deriv = ip.weight; //dR/dWeight
        isotrans.WeightRevDiff(PointMat_bar);        
        el->CalcShape(ip, shape);
        if (Q)
        {
            deriv *= Q->Eval(*Tr, ip);
        }

        // perform deriv*(adj*shape)*(shape*elfun)
        double rw = deriv*(eladj*shape)*(shape*elfun);
        PointMat_bar.Set(rw, PointMat_bar); //dWeight/dX

        for (int j = 0; j < dof ; ++j)
        {
            for (int d = 0; d < dim; ++d)
            {
                elvect(d*dof + j) += PointMat_bar(d,j);
            }
        }
    }
}

void DiffusionResIntegrator::AssembleElementVector(const FiniteElement &elx,
                                       ElementTransformation &Trx,
                                       const Vector &elfunx, Vector &elvect)
{
    /// get the proper element, transformation, and state vector
    Array<int> vdofs; Vector elfun; Vector eladj; 
    int element = Trx.ElementNo;
    const FiniteElement *el = state->FESpace()->GetFE(element);
    ElementTransformation *Tr = state->FESpace()->GetElementTransformation(element);
    state->FESpace()->GetElementVDofs(element, vdofs);
    int order = 2*el->GetOrder() + Tr->OrderW();
    const IntegrationRule *ir = NULL;
    if (ir == NULL)
    {
        ir = &IntRules.Get(el->GetGeomType(), order);
    }
    state->GetSubVector(vdofs, elfun);
    adjoint->GetSubVector(vdofs, eladj);

    const int dof = elx.GetDof();
    const int dofu = el->GetDof();
    const int dim = el->GetDim();
    int spaceDim = Tr->GetSpaceDim();
    bool square = (dim == spaceDim);
    double tw, w, dw;
    Vector av(dim); Vector bv(dim);
    Vector ad(dim); Vector bd(dim);
    elvect.SetSize(dof*dim);
    elvect = 0.0;

    // cast the ElementTransformation
    IsoparametricTransformation &isotrans =
    dynamic_cast<IsoparametricTransformation&>(*Tr);

    DenseMatrix elmat(dofu);
    DenseMatrix PointMat_bar(dim, dof);
    DenseMatrix jac_bar(dim);
    dshape.SetSize(dofu, dim);

    // loop through nodes
    for (int i = 0; i < ir->GetNPoints(); ++i)
    {
        const IntegrationPoint &ip = ir->IntPoint(i);
        Tr->SetIntPoint(&ip);

        DenseMatrix K = Tr->AdjugateJacobian();
        PointMat_bar = 0.0;
        jac_bar = 0.0;

        el->CalcDShape(ip, dshape);
        tw = Tr->Weight();
        w = ip.weight / (square ? tw : tw*tw*tw);
        dw = -ip.weight / (square ? tw*tw : tw*tw*tw*tw/3);
        /// NOTE: Q may or may not have sensitivity to x. Need to tailor to
        /// different coefficients 
        if (Q)
        {
            w *= Q->Eval(*Tr, ip);
            dw *= Q->Eval(*Tr, ip);
        }
        dshape.MultTranspose(eladj, ad); //D^T\psi
        K.MultTranspose(ad, av); //K^TD^T\psi
        dshape.MultTranspose(elfun, bd); //D^T\u
        K.MultTranspose(bd, bv); //K^TD^T\u

        // compute partials wrt weight
        double rw = dw*(av*bv);
        isotrans.WeightRevDiff(PointMat_bar);
        PointMat_bar.Set(rw, PointMat_bar);

        // compute partials wrt adjugate
        AddMult_a_VWt(w, ad, bv, jac_bar);
        AddMult_a_VWt(w, bd, av, jac_bar);
        isotrans.AdjugateJacobianRevDiff(jac_bar, PointMat_bar);
        
        for (int j = 0; j < dof ; ++j)
        {
            for (int d = 0; d < dim; ++d)
            {
                elvect(d*dof + j) += PointMat_bar(d,j);
            }
        }
    }
}

void DiffusionResIntegrator::AssembleRHSElementVect(
    const mfem::FiniteElement &elx,
    mfem::ElementTransformation &Trx,
    mfem::Vector &elvect)
{
    /// get the proper element, transformation, and state vector
    Array<int> vdofs; Vector elfun; Vector eladj; 
    int element = Trx.ElementNo;
    const FiniteElement *el = state->FESpace()->GetFE(element);
    ElementTransformation *Tr = state->FESpace()->GetElementTransformation(element);
    state->FESpace()->GetElementVDofs(element, vdofs);
    int order = 2*el->GetOrder() + Tr->OrderW();
    const IntegrationRule *ir = NULL;
    if (ir == NULL)
    {
        ir = &IntRules.Get(el->GetGeomType(), order);
    }
    state->GetSubVector(vdofs, elfun);
    adjoint->GetSubVector(vdofs, eladj);

    const int dof = elx.GetDof();
    const int dofu = el->GetDof();
    const int dim = el->GetDim();
    int spaceDim = Tr->GetSpaceDim();
    bool square = (dim == spaceDim);
    double tw, w, dw;
    Vector av(dim); Vector bv(dim);
    Vector ad(dim); Vector bd(dim);
    elvect.SetSize(dof*dim);
    elvect = 0.0;

    // cast the ElementTransformation
    IsoparametricTransformation &isotrans =
    dynamic_cast<IsoparametricTransformation&>(*Tr);

    DenseMatrix elmat(dofu);
    DenseMatrix PointMat_bar(dim, dof);
    DenseMatrix jac_bar(dim);
    dshape.SetSize(dofu, dim);

    // loop through nodes
    for (int i = 0; i < ir->GetNPoints(); ++i)
    {
        const IntegrationPoint &ip = ir->IntPoint(i);
        Tr->SetIntPoint(&ip);

        DenseMatrix K = Tr->AdjugateJacobian();
        PointMat_bar = 0.0;
        jac_bar = 0.0;

        el->CalcDShape(ip, dshape);
        tw = Tr->Weight();
        w = ip.weight / (square ? tw : tw*tw*tw);
        dw = -ip.weight / (square ? tw*tw : tw*tw*tw*tw/3);
        /// NOTE: Q may or may not have sensitivity to x. Need to tailor to
        /// different coefficients 
        if (Q)
        {
            w *= Q->Eval(*Tr, ip);
            dw *= Q->Eval(*Tr, ip);
        }
        dshape.MultTranspose(eladj, ad); //D^T\psi
        K.MultTranspose(ad, av); //K^TD^T\psi
        dshape.MultTranspose(elfun, bd); //D^T\u
        K.MultTranspose(bd, bv); //K^TD^T\u

        // compute partials wrt weight
        double rw = dw*(av*bv);
        isotrans.WeightRevDiff(PointMat_bar);
        PointMat_bar.Set(rw, PointMat_bar);

        // compute partials wrt adjugate
        AddMult_a_VWt(w, ad, bv, jac_bar);
        AddMult_a_VWt(w, bd, av, jac_bar);
        isotrans.AdjugateJacobianRevDiff(jac_bar, PointMat_bar);
        
        for (int j = 0; j < dof ; ++j)
        {
            for (int d = 0; d < dim; ++d)
            {
                elvect(d*dof + j) += PointMat_bar(d,j);
            }
        }
    }
}


void BoundaryNormalResIntegrator::AssembleRHSElementVect(
                                        const FiniteElement &elx,
                                         FaceElementTransformations &Trx,
                                         Vector &elvect)
{   
    /// get the proper element, transformation, and state vector
    Array<int> vdofs; Vector elfun; Vector eladj;
    int element = Trx.Elem1No;
    const FiniteElementCollection *fec = state->FESpace()->FEColl();
    const FiniteElement *el = state->FESpace()->GetFE(element);

    const int dof = elx.GetDof();
    const int dofu = el->GetDof();
    // const int dim = Trx.Face->GetDimension();
    int space_dim = Trx.Face->GetSpaceDim();
    shape.SetSize(dofu);
    elvect.SetSize(space_dim*dof);
    elvect = 0.0;

    // get the right boundary element
    const FiniteElement *el_bnd;
    switch (space_dim)
    {
        case 1: el_bnd = fec->FiniteElementForGeometry(Geometry::POINT);
              break;
        case 2: el_bnd = fec->FiniteElementForGeometry(Geometry::SEGMENT);
              break;
        case 3:
            if(Trx.Elem1->GetGeometryType() == Geometry::TETRAHEDRON)
            {
                el_bnd = fec->FiniteElementForGeometry(Geometry::TRIANGLE);
            }
            if(Trx.Elem1->GetGeometryType() == Geometry::CUBE)
            {
                el_bnd = fec->FiniteElementForGeometry(Geometry::SQUARE);
            } 
             break;
    }
    ElementTransformation *Tr_bnd = Trx.Face;
    ElementTransformation *Tr = Trx.Elem1;
    
    //boundary element integration rule
    const IntegrationRule *ir = IntRule;
    if (ir == NULL)
    {
        ir = &IntRules.Get(el_bnd->GetGeomType(), oa * el_bnd->GetOrder() + ob);
    }

    state->FESpace()->GetElementVDofs(element, vdofs);
    state->GetSubVector(vdofs, elfun); //don't need this one
    adjoint->GetSubVector(vdofs, eladj);

    // cast the ElementTransformation (for the domain element)
    IsoparametricTransformation &isotrans =
    dynamic_cast<IsoparametricTransformation&>(*Trx.Elem1);

    DenseMatrix PointMat_bar(space_dim, dof);
    DenseMatrix R;
    Vector Qvec(space_dim);
    
    // loop through nodes
    for (int i = 0; i < ir->GetNPoints(); ++i)
    {
        //compute dR/dnor*dnor/dJ_bnd*dJ_bnd/dJ_el*dJ_el/dX

        //get corresponding element integration point
        const IntegrationPoint &ip = ir->IntPoint(i);
        IntegrationPoint eip;
        Trx.Loc1.Transform(ip, eip);
        Tr_bnd->SetIntPoint(&ip);
        Tr->SetIntPoint(&eip);

        //get el to bnd jacobian transformation
        DenseMatrix J = Tr_bnd->Jacobian();
        DenseMatrix Jinv_el = Tr->InverseJacobian();
        R.SetSize(Jinv_el.Height(), J.Width());
        Mult(Jinv_el, J, R);
        DenseMatrix J_bar(J.Height(), J.Width());
        DenseMatrix J_bar_el(space_dim, space_dim);

        PointMat_bar = 0.0;

        /// NOTE: Q may or may not have sensitivity to x. Need to tailor to
        /// different coefficients 
        el->CalcShape(eip, shape);
        Q.Eval(Qvec, *Tr_bnd, ip);
        Vector nor_bar(Qvec.Size()); 
        for (int p = 0; p < nor_bar.Size(); ++p) //dR/dnor
        {
            nor_bar(p) = ip.weight*Qvec(p)*(eladj*shape); 
        }
        CalcOrthoRevDiff(J, nor_bar, J_bar); //dnor/dJbnd

        //convert face jacobian bar to element jacobian bar by inverting
        MultABt(J_bar, R, J_bar_el); //dJbnd/dJel

        isotrans.JacobianRevDiff(J_bar_el, PointMat_bar); //dJel/dX

        for (int j = 0; j < dof ; ++j)
        {
            for (int d = 0; d < space_dim; ++d)
            {
                elvect(d* dof + j) += PointMat_bar (d,j);
            }
        }
    }
}


#if 0
double DomainResIntegrator::GetElementEnergy(const FiniteElement &elx,
                                       ElementTransformation &Trx,
                                       const Vector &elfunx)
{
    double Rpart = 0;
    
    /// get the proper element, transformation, and state vector
    Array<int> vdofs; Vector elfun; Vector eladj;
    int element = Trx.ElementNo;
    const FiniteElement *el = state->FESpace()->GetFE(element);
    ElementTransformation *Tr = state->FESpace()->GetElementTransformation(element);
    state->FESpace()->GetElementVDofs(element, vdofs);
    const IntegrationRule *ir = IntRule;
    if (ir == NULL)
    {
        ir = &IntRules.Get(el->GetGeomType(), oa * el->GetOrder() + ob);
    }
    state->GetValues(element, *ir, elfun); //don't need this one
    adjoint->GetValues(element, *ir, eladj);

    const int dof = el->GetDof();
    const int dim = el->GetDim();
    
    Vector x_q(dim);
    DenseMatrix Jac_q(dim, dim);
    
    // loop through nodes
    for (int i = 0; i < ir->GetNPoints(); ++i)
    {
        const IntegrationPoint &ip = ir->IntPoint(i);
        Tr->SetIntPoint(&ip);
        Tr->Transform(ip, x_q);
        Jac_q = Tr->Jacobian();
        double r_q = Tr->Weight()*Q.Eval(*Tr, ip);
        //double r_q = calcFunctional(Tr->ElementNo,
        //                             ip, x_q, Tr, Jac_q);
        
        //skipping shape function step
        Rpart += ip.weight*r_q;
    }

   return Rpart;
}

double MassResIntegrator::GetElementEnergy(const FiniteElement &elx,
                                       ElementTransformation &Trx,
                                       const Vector &elfunx)
{
    double Rpart = 0;
    
    /// get the proper element, transformation, and state vector
    Array<int> vdofs; Vector elfun; Vector eladj;
    int element = Trx.ElementNo;
    const FiniteElement *el = state->FESpace()->GetFE(element);
    ElementTransformation *Tr = state->FESpace()->GetElementTransformation(element);
    state->FESpace()->GetElementVDofs(element, vdofs);
    int order = 2*el->GetOrder() + Tr->OrderW();
    const IntegrationRule *ir = IntRule;
    if (ir == NULL)
    {
        ir = &IntRules.Get(el->GetGeomType(), order);
    }
    state->GetValues(element, *ir, elfun); 
    adjoint->GetValues(element, *ir, eladj);

    Array<int> dofs;
    const int dof = el->GetDof();
    const int dim = el->GetDim();
    
    Vector x_q(dim);
    Vector rvect(dof);
    DenseMatrix Jac_q(dim, dim);
    DenseMatrix elmat(dof);
    
    // assemble the matrix
    elmat = 0.0;
    for (int i = 0; i < ir->GetNPoints(); ++i)
    {
        const IntegrationPoint &ip = ir->IntPoint(i);
        el->CalcShape(ip, shape);

        Tr->SetIntPoint (&ip);
        double w = Tr->Weight() * ip.weight;
        if (Q)
        {
            w *= Q->Eval(*Tr, ip);
        }

        AddMult_a_VVt(w, shape, elmat);
    }

    elmat.Mult(elfun, rvect);

    return rvect.Sum();
}

#endif

}


       