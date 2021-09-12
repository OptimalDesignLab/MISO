template <typename Derived>
void MMSIntegrator<Derived>::AssembleElementVector(
    const mfem::FiniteElement &el, mfem::ElementTransformation &trans,
    const mfem::Vector &elfun, mfem::Vector &elvect)
{
   using namespace mfem;
   const SBPFiniteElement &sbp = dynamic_cast<const SBPFiniteElement&>(el);
   // const IntegrationRule& ir = sbp.GetNodes();
   int num_nodes = sbp.GetDof();
   // int dim = sbp.GetDim();
#ifdef MFEM_THREAD_SAFE
   Vector x_i, src_i;
#endif
   src_i.SetSize(num_states);
	elvect.SetSize(num_states*num_nodes);
   DenseMatrix res(elvect.GetData(), num_nodes, num_states);
   elvect = 0.0;
   for (int i = 0; i < num_nodes; ++i)
   {
      const IntegrationPoint &ip = el.GetNodes().IntPoint(i);
      trans.SetIntPoint(&ip);
      trans.Transform(ip, x_i);
      double weight = trans.Weight()*ip.weight;
      source(x_i, src_i);
      for (int n = 0; n < num_states; ++n)
      {
         res(i, n) += weight*src_i(n);
      }
   }
   res *= alpha;
}

template <typename Derived>
void MMSIntegrator<Derived>::AssembleElementGrad(
    const mfem::FiniteElement &el, mfem::ElementTransformation &trans,
    const mfem::Vector &elfun, mfem::DenseMatrix &elmat)
{
   using namespace mfem;
   const SBPFiniteElement &sbp = dynamic_cast<const SBPFiniteElement&>(el);
   // const IntegrationRule& ir = sbp.GetNodes();
   int num_nodes = sbp.GetDof();
   elmat.SetSize(num_states*num_nodes);
   elmat = 0.0;
}