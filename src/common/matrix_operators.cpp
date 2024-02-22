#include "mfem.hpp"

#include "matrix_operators.hpp"

using namespace std;
using namespace mfem;

namespace miso
{
SumOfOperators::SumOfOperators(double alpha,
                               Operator &oper1,
                               double beta,
                               Operator &oper2)
 : Operator(oper1.Height(), oper1.Width()),
   a(alpha),
   b(beta),
   oper_a(&oper1),
   oper_b(&oper2),
   work_vec(oper1.Width())
{
   if ((oper_a->Height() != oper_b->Height()) ||
       (oper_a->Width() != oper_b->Width()))
   {
      throw MISOException("SumOfOperators: Operator sizes are incompatible!\n");
   }
}

void SumOfOperators::Add(double alpha,
                         Operator &oper1,
                         double beta,
                         Operator &oper2)
{
   a = alpha;
   b = beta;
   oper_a = &oper1;
   oper_b = &oper2;
}

void SumOfOperators::Mult(const Vector &x, Vector &y) const
{
   y.SetSize(x.Size());
   const double zero = 1e-16;
   if (fabs(a) > zero)
   {
      oper_a->Mult(x, work_vec);
      if (fabs(a - 1.0) > zero)
      {
         work_vec *= a;
      }
   }
   if (fabs(b) > zero)
   {
      oper_b->Mult(x, y);
      if (fabs(b - 1.0) > zero)
      {
         y *= b;
      }
   }
   y += work_vec;
}

JacobianFree::JacobianFree(MISOResidual &residual)
 : Operator(getSize(residual)),
   comm(getMPIComm(residual)),
   scale(1.0),
   res(residual),
   explicit_part(nullptr),
   state(getSize(res)),
   res_at_state(getSize(res)),
   state_pert(getSize(res))
{ }

void JacobianFree::setState(const mfem::Vector &baseline)
{
   // store baseline state, because we need to perturb it later
   state = baseline;
   // initialize the res_at_state vector for later use
   auto inputs = MISOInputs({{"state", baseline}});
   evaluate(res, inputs, res_at_state);
}

void JacobianFree::setState(const MISOInputs &inputs)
{
   // store baseline state, because we need to perturb it later
   setVectorFromInputs(inputs, "state", state, true, true);
   evaluate(res, inputs, res_at_state);
}

void JacobianFree::Mult(const mfem::Vector &x, mfem::Vector &y) const
{
   if (fabs(scale) > zero)
   {
      // apply the Jacobian-free operator
      double eps_fd = getStepSize(state, x);
      // create the perturbed vector, and evaluate the residual
      add(state, eps_fd, x, state_pert);
      auto inputs = MISOInputs({{"state", state_pert}});
      evaluate(res, inputs, y);
      // subtract the baseline residual and divide by eps_fd to get product
      subtract(1 / eps_fd, y, res_at_state, y);
      if (fabs(scale - 1.0) > zero)
      {
         y *= scale;
      }
   }
   if (explicit_part)
   {
      // Include contribution from explicit operator, if necessary
      explicit_part->Mult(x, state_pert);
      y += state_pert;
   }
}

mfem::Operator &JacobianFree::getDiagonalBlock(int i) const
{
   // First, extract the Jacobian block and cast the explicit part
   // We assume that `state` holds where the Jacobian is to be evaluated
   auto inputs = MISOInputs({{"state", state}});
   Operator &jac = getJacobianBlock(res, inputs, i);
   BlockOperator *block_op = dynamic_cast<BlockOperator *>(explicit_part);
   if (explicit_part != nullptr && block_op == nullptr)
   {
      throw MISOException(
          "JacobianFree::getDiagonalBlock:\n"
          "explicit part of operator must be castable to "
          "BlockOperator!\n");
   }

   // Case 1: HypreParMatrix
   HypreParMatrix *hypre_jac = dynamic_cast<HypreParMatrix *>(&jac);
   if (hypre_jac)
   {
      *hypre_jac *= scale;
      if (block_op)
      {
         Operator &exp_block = block_op->GetBlock(i, i);
         HypreParMatrix *hypre_exp = dynamic_cast<HypreParMatrix *>(&exp_block);
         *hypre_jac += *hypre_exp;
      }
      return jac;
   }
   // Case 2: DenseMatrix
   DenseMatrix *dense_jac = dynamic_cast<DenseMatrix *>(&jac);
   if (dense_jac)
   {
      *dense_jac *= scale;
      if (block_op)
      {
         Operator &exp_block = block_op->GetBlock(i, i);
         DenseMatrix *dense_exp = dynamic_cast<DenseMatrix *>(&exp_block);
         *dense_jac += *dense_exp;
      }
      return jac;
   }
   // If we get here, there was some kind of problem
   throw MISOException(
       "JacobianFree::getDiagonalBlock:\n"
       "incompatible operators/matrices!\n");
}

double JacobianFree::getStepSize(const mfem::Vector &baseline,
                                 const mfem::Vector &pert) const
{
   // This is based on the step size suggested by Chisholm and Zingg in
   // "A Jacobian-Free Newton-Krylov Algorithm for Compressible Turbulent Fluid
   // Flows", https://doi.org/10.1016/j.jcp.2009.02.004
   const double delta = 1e-10;
   double prod = InnerProduct(comm, pert, pert);
   if (prod > 1e-6)
   {
      return sqrt(delta / prod);
   }
   else
   {
      return 1e-7;
   }
}

void JacobianFree::print(string file_name) const
{
   remove(file_name.c_str());
   ofstream matrix_file(file_name, fstream::app);
   matrix_file << setprecision(16);

   Vector y(state.Size());
   Vector work(state.Size());
   double eps_fd = getStepSize(state, state);
   cout << "eps_fd " << eps_fd << endl;
   for (int j = 0; j < width; ++j)
   {
      // perturb state in jth variable
      state_pert = state;
      state_pert(j) += eps_fd;
      auto inputs = MISOInputs({{"state", state_pert}});
      evaluate(res, inputs, y);
      // subtract the baseline residual and divide by eps_fd to get product
      subtract(1 / eps_fd, y, res_at_state, y);
      if (fabs(scale - 1.0) > zero)
      {
         y *= scale;
      }
      if (explicit_part)
      {
         state_pert = 0.0;
         state_pert(j) = 1.0;
         explicit_part->Mult(state_pert, work);
         y += work;
      }
      for (int i = 0; i < height; ++i)
      {
         matrix_file << y(i);
         if (i != height - 1)
         {
            matrix_file << ", ";
         }
         else
         {
            matrix_file << endl;
         }
      }
   }
   matrix_file.close();
}

}  // namespace miso