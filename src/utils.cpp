#include "utils.hpp"

using namespace mfem;
using namespace std;

namespace mach
{

/// performs the Hadamard (elementwise) product: `v(i) = v1(i)*v2(i)`
void multiplyElementwise(const Vector &v1, const Vector &v2, Vector &v)
{
   MFEM_ASSERT(v1.Size() == v2.Size() && v1.Size() == v.Size(), "");
   for (int i = 0; i < v.Size(); ++i)
   {
      v(i) = v1(i) * v2(i);
   }
}

/// performs the Hadamard (elementwise) product: `a(i) *= b(i)`
void multiplyElementwise(const Vector &b, Vector &a)
{
   MFEM_ASSERT( a.Size() == b.Size(), "");
   for (int i = 0; i < a.Size(); ++i)
   {
      a(i) *= b(i);
   }
}

/// performs an elementwise division: `v(i) = v1(i)/v2(i)`
void divideElementwise(const Vector &v1, const Vector &v2, Vector &v)
{
   MFEM_ASSERT(v1.Size() == v2.Size() && v1.Size() == v.Size(), "");
   for (int i = 0; i < v.Size(); ++i)
   {
      v(i) = v1(i) / v2(i);
   }
}

/// performs elementwise inversion: `y(i) = 1/x(i)`
void invertElementwise(const Vector &x, Vector &y)
{
   MFEM_ASSERT(x.Size() == y.Size(), "");
   for (int i = 0; i < x.Size(); ++i)
   {
      y(i) = 1.0 / x(i);
   }
}

/// performs quadratic interpolation given x0, y0, dy0/dx0, x1, and y1.
double quadInterp(double x0, double y0, double dydx0, double x1, double y1)
{
   // Assume the fuction has the form y(x) = c0 + c1 * x + c2 * x^2
   double c0, c1, c2;
   c0 = (dydx0 * x0 * x0 * x1 + y1 * x0 * x0 - dydx0 * x0 * x1 * x1 - 2 * y0 * x0 * x1 + y0 * x1 * x1) /
        (x0 * x0 - 2 * x1 * x0 + x1 * x1);
   c1 = (2 * x0 * y0 - 2 * x0 * y1 - x0 * x0 * dydx0 + x1 * x1 * dydx0) /
        (x0 * x0 - 2 * x1 * x0 + x1 * x1);
   c2 = -(y0 - y1 - x0 * dydx0 + x1 * dydx0) / (x0 * x0 - 2 * x1 * x0 + x1 * x1);
   return -c1 / (2 * c2);
}

DiscreteInterpolationOperator::~DiscreteInterpolationOperator()
{}

DiscreteGradOperator::DiscreteGradOperator(SpaceType *dfes,
                                           SpaceType *rfes)
   : DiscreteInterpolationOperator(dfes, rfes)
{
   this->AddDomainInterpolator(new GradientInterpolator);
}

DiscreteCurlOperator::DiscreteCurlOperator(SpaceType *dfes,
                                           SpaceType *rfes)
   : DiscreteInterpolationOperator(dfes, rfes)
{
   this->AddDomainInterpolator(new CurlInterpolator);
}

DiscreteDivOperator::DiscreteDivOperator(SpaceType *dfes,
                                         SpaceType *rfes)
   : DiscreteInterpolationOperator(dfes, rfes)
{
   this->AddDomainInterpolator(new DivergenceInterpolator);
}

IrrotationalProjector
::IrrotationalProjector(SpaceType &H1FESpace,
                        SpaceType &HCurlFESpace,
                        const int &irOrder,
                        BilinearFormType *s0,
                        MixedBilinearFormType *weakDiv,
                        DiscreteGradOperator *grad)
   : H1FESpace_(&H1FESpace),
     HCurlFESpace_(&HCurlFESpace),
     s0_(s0),
     weakDiv_(weakDiv),
     grad_(grad),
     psi_(NULL),
     xDiv_(NULL),
     S0_(NULL),
     amg_(NULL),
     pcg_(NULL),
     ownsS0_(s0 == NULL),
     ownsWeakDiv_(weakDiv == NULL),
     ownsGrad_(grad == NULL)
{
   /// not sure if theres a better way to handle this
#ifdef MFEM_USE_MPI
   ess_bdr_.SetSize(H1FESpace_->GetParMesh()->bdr_attributes.Max());
#else
   ess_bdr_.SetSize(H1FESpace_->GetMesh()->bdr_attributes.Max());
#endif
   ess_bdr_ = 1;
   H1FESpace_->GetEssentialTrueDofs(ess_bdr_, ess_bdr_tdofs_);

   int geom = H1FESpace_->GetFE(0)->GetGeomType();
   const IntegrationRule * ir = &IntRules.Get(geom, irOrder);

   if ( s0 == NULL )
   {
      s0_ = new BilinearFormType(H1FESpace_);
      BilinearFormIntegrator *diffInteg = new DiffusionIntegrator;
      diffInteg->SetIntRule(ir);
      s0_->AddDomainIntegrator(diffInteg);
      s0_->Assemble();
      s0_->Finalize();
      S0_ = new MatrixType;
   }
   if ( weakDiv_ == NULL )
   {
      weakDiv_ = new MixedBilinearFormType(HCurlFESpace_, H1FESpace_);
      BilinearFormIntegrator *wdivInteg = new VectorFEWeakDivergenceIntegrator;
      wdivInteg->SetIntRule(ir);
      weakDiv_->AddDomainIntegrator(wdivInteg);
      weakDiv_->Assemble();
      weakDiv_->Finalize();
   }
   if ( grad_ == NULL )
   {
      grad_ = new DiscreteGradOperator(H1FESpace_, HCurlFESpace_);
      grad_->Assemble();
      grad_->Finalize();
   }

   psi_  = new GridFunType(H1FESpace_);
   xDiv_ = new GridFunType(H1FESpace_);
}

IrrotationalProjector::~IrrotationalProjector()
{
   delete psi_;
   delete xDiv_;

#ifdef MFEM_USE_MPI
   delete amg_;
   delete pcg_;
#endif

   delete S0_;

   delete s0_;
   delete weakDiv_;
}

void
IrrotationalProjector::InitSolver() const
{

   delete pcg_;
   delete amg_;

#ifdef MFEM_USE_MPI
   amg_ = new HypreBoomerAMG(*S0_);
   amg_->SetPrintLevel(0);
   pcg_ = new HyprePCG(*S0_);
   pcg_->SetTol(1e-14);
   pcg_->SetMaxIter(200);
   pcg_->SetPrintLevel(0);
   pcg_->SetPreconditioner(*amg_);
#else
   amg_ = new EMPrecType((SparseMatrix&)(*S0_));

   // CGSolver pcg_;
   pcg_ = new CGSolver();
   pcg_->SetPrintLevel(1);
   pcg_->SetMaxIter(400);
   pcg_->SetRelTol(1e-14);
   pcg_->SetAbsTol(1e-14);
   pcg_->SetPreconditioner(*amg_);
   pcg_->SetOperator(*S0_);
#endif
}

void
IrrotationalProjector::Mult(const Vector &x, Vector &y) const
{
   // Compute the divergence of x
   weakDiv_->Mult(x,*xDiv_); *xDiv_ *= -1.0;
   std::cout << "weakdiv mult\n";

   // Apply essential BC and form linear system
   *psi_ = 0.0;
   std::cout <<"psi length: "<< psi_->Size() <<"\n xDiv length: "<< xDiv_->Size() <<"\n Psi length: "<< Psi_.Size() <<"\n RHS length: "<< RHS_.Size() <<"\n";
   s0_->FormLinearSystem(ess_bdr_tdofs_, *psi_, *xDiv_, *S0_, Psi_, RHS_);
   std::cout << "form lin system\n";

   // Solve the linear system for Psi
   if ( pcg_ == NULL ) { this->InitSolver(); }
   pcg_->Mult(RHS_, Psi_);
   std::cout << "pcg mult\n";

   // Compute the parallel grid function correspoinding to Psi
   s0_->RecoverFEMSolution(Psi_, *xDiv_, *psi_);

   // Compute the irrotational portion of x
   grad_->Mult(*psi_, y);
}

void
IrrotationalProjector::Update()
{
   delete pcg_; pcg_ = NULL;
   delete amg_; amg_ = NULL;
   delete S0_;  S0_  = new MatrixType;

   psi_->Update();
   xDiv_->Update();

   if ( ownsS0_ )
   {
      s0_->Update();
      s0_->Assemble();
      s0_->Finalize();
   }
   if ( ownsWeakDiv_ )
   {
      weakDiv_->Update();
      weakDiv_->Assemble();
      weakDiv_->Finalize();
   }
   if ( ownsGrad_ )
   {
      grad_->Update();
      grad_->Assemble();
      grad_->Finalize();
   }

   H1FESpace_->GetEssentialTrueDofs(ess_bdr_, ess_bdr_tdofs_);
}

DivergenceFreeProjector
::DivergenceFreeProjector(SpaceType &H1FESpace,
                          SpaceType &HCurlFESpace,
                          const int &irOrder,
                          BilinearFormType *s0,
                          MixedBilinearFormType *weakDiv,
                          DiscreteGradOperator *grad)
   : IrrotationalProjector(H1FESpace,HCurlFESpace, irOrder, s0, weakDiv, grad)
{}

void DivergenceFreeProjector::Mult(const Vector &x, Vector &y) const
{
   std::cout << "above irrot proj mult\n";
   this->IrrotationalProjector::Mult(x, y);
   std::cout << "below irrot proj mult\n";
   y  -= x;
   y *= -1.0;
}

void DivergenceFreeProjector::Update()
{
   this->IrrotationalProjector::Update();
}

#ifndef MFEM_USE_LAPACK
void dgelss_(int *, int *, int *, double *, int *, double *, int *, double *,
        double *, int *, double *, int *, int *);
void dgels_(char *, int *, int *, int *, double *, int *, double *, int *, double *,
       int *, int *);
#else
extern "C" void
dgelss_(int *, int *, int *, double *, int *, double *, int *, double *,
        double *, int *, double *, int *, int *);
extern "C" void
dgels_(char *, int *, int *, int *, double *, int *, double *, int *, double *,
       int *, int *);
#endif
/// build the interpolation operator on element patch
/// this function will be moved later
#ifdef MFEM_USE_LAPACK
void buildInterpolation(int degree, const DenseMatrix &x_center,
                        const DenseMatrix &x_quad, DenseMatrix &interp)
{
   // number of quadrature points
   int num_quad = x_quad.Width();
   // number of elements
   int num_el = x_center.Width();

   // number of row and colomn in little r matrix
   int m = (degree + 1) * (degree + 2) / 2;
   int n = num_el;

   // Set the size of interpolation operator
   interp.SetSize(num_quad, num_el);
   // required by the lapack routine
   mfem::DenseMatrix rhs(n, 1);
   char TRANS = 'N';
   int nrhs = 1;
   int lwork = 2 * m * n;
   double work[lwork];

   // construct each row of R (also loop over each quadrature point)
   for (int i = 0; i < num_quad; i++)
   {
      // reset the rhs
      rhs = 0.0;
      rhs(0, 0) = 1.0;
      // construct the aux matrix to solve each row of R
      DenseMatrix r(m, n);
      r = 0.0;
      // loop over each column of r
      for (int j = 0; j < n; j++)
      {
         double x_diff = x_center(0, j) - x_quad(0, i);
         double y_diff = x_center(1, j) - x_quad(1, i);
         r(0, j) = 1.0;
         int index = 1;
         // loop over different orders
         for (int order = 1; order <= degree; order++)
         {
            for (int c = order; c >= 0; c--)
            {
               r(index, j) = pow(x_diff, c) * pow(y_diff, order - c);
               index++;
            }
         }
      }
      // Solve each row of R and put them back to R
      int info, rank;
      dgels_(&TRANS, &m, &n, &nrhs, r.GetData(), &m, rhs.GetData(), &n,
             work, &lwork, &info);
      MFEM_ASSERT(info == 0, "Fail to solve the underdetermined system.\n");
      // put each row back to interp
      for (int k = 0; k < n; k++)
      {
         interp(i, k) = rhs(k, 0);
      }
   }
} // end of constructing interp
#endif

} // namespace mach
