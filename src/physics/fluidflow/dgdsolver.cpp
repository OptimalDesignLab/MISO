#include <memory>
#include "dgdsolver.hpp"
#include "euler_integ_DG.hpp"
using namespace mfem;
using namespace std;

namespace mach
{

    template <int dim, bool entvar>
    DGDSolver<dim, entvar>::DGDSolver(
        const nlohmann::json &json_options,
        unique_ptr<mfem::Mesh> smesh,
        MPI_Comm comm)
        : AbstractSolver(json_options, move(smesh), comm)
    {
        if (entvar)
        {
            *out << "The state variables are the entropy variables." << endl;
        }
        else
        {
            *out << "The state variables are the conservative variables." << endl;
        }
        // define free-stream parameters; may or may not be used, depending on case
        mach_fs = options["flow-param"]["mach"].template get<double>();
        aoa_fs = options["flow-param"]["aoa"].template get<double>() * M_PI / 180;
        iroll = options["flow-param"]["roll-axis"].template get<int>();
        ipitch = options["flow-param"]["pitch-axis"].template get<int>();
        if (iroll == ipitch)
        {
            throw MachException("iroll and ipitch must be distinct dimensions!");
        }
        if ((iroll < 0) || (iroll > 2))
        {
            throw MachException("iroll axis must be between 0 and 2!");
        }
        if ((ipitch < 0) || (ipitch > 2))
        {
            throw MachException("ipitch axis must be between 0 and 2!");
        }
    }

    template <int dim, bool entvar>
    void DGDSolver<dim, entvar>::constructForms()
    {
        res.reset(new NonlinearFormType(fes_GD.get()));
        if ((entvar) && (!options["time-dis"]["steady"].template get<bool>()))
        {
            nonlinear_mass.reset(new NonlinearFormType(fes_GD.get()));
            mass.reset();
        }
        else
        {
            mass.reset(new BilinearFormType(fes_GD.get()));
            nonlinear_mass.reset();
        }
        ent.reset(new NonlinearFormType(fes_GD.get()));
    }

    template <int dim, bool entvar>
    void DGDSolver<dim, entvar>::addMassIntegrators(double alpha)
    {
        mass->AddDomainIntegrator(new EulerMassIntegrator(num_state));
    }

    template <int dim, bool entvar>
    void DGDSolver<dim, entvar>::addNonlinearMassIntegrators(double alpha)
    {
        nonlinear_mass->AddDomainIntegrator(
            new MassIntegrator<dim, entvar>(diff_stack, alpha));
    }

    template <int dim, bool entvar>
    void DGDSolver<dim, entvar>::addResVolumeIntegrators(double alpha)
    {
        // TODO: should decide between one-point and two-point fluxes using options
        res->AddDomainIntegrator(
            new EulerDomainIntegrator<2>(diff_stack, num_state, alpha));
    }

    template <int dim, bool entvar>
    void DGDSolver<dim, entvar>::addResBoundaryIntegrators(double alpha)
    {
        auto &bcs = options["bcs"];
        int idx = 0;
        if (bcs.find("vortex") != bcs.end())
        { // slip-wall boundary condition
            vector<int> tmp = bcs["vortex"].template get<vector<int>>();
            bndry_marker[idx].SetSize(tmp.size(), 0);
            bndry_marker[idx].Assign(tmp.data());
            mfem::Vector qfar(dim + 2);
            getFreeStreamState(qfar);
            res->AddBdrFaceIntegrator(
                new EulerBoundaryIntegrator<dim, 1, entvar>(diff_stack, fec.get(),
                                                            num_state, qfar, alpha),
                bndry_marker[idx]);
            idx++;
        }
        if (bcs.find("slip-wall") != bcs.end())
        { // slip-wall boundary condition
            vector<int> tmp = bcs["slip-wall"].template get<vector<int>>();
            bndry_marker[idx].SetSize(tmp.size(), 0);
            bndry_marker[idx].Assign(tmp.data());
            mfem::Vector qfar(dim + 2);
            getFreeStreamState(qfar);
            res->AddBdrFaceIntegrator(
                new EulerBoundaryIntegrator<dim, 2, entvar>(diff_stack, fec.get(),
                                                            num_state, qfar, alpha),
                bndry_marker[idx]);
            idx++;
        }
        if (bcs.find("far-field") != bcs.end())
        {
            // far-field boundary conditions
            vector<int> tmp = bcs["far-field"].template get<vector<int>>();
            mfem::Vector qfar(dim + 2);
            getFreeStreamState(qfar);
            bndry_marker[idx].SetSize(tmp.size(), 0);
            bndry_marker[idx].Assign(tmp.data());
            res->AddBdrFaceIntegrator(
                new EulerBoundaryIntegrator<dim, 3, entvar>(diff_stack, fec.get(),
                                                            num_state, qfar, alpha),
                bndry_marker[idx]);
            idx++;
        }
    }

    template <int dim, bool entvar>
    void DGDSolver<dim, entvar>::addResInterfaceIntegrators(double alpha)
    {
        // add the integrators based on if discretization is continuous or discrete
        if (options["space-dis"]["basis-type"].template get<string>() == "DG")
        {
            res->AddInteriorFaceIntegrator(
                new EulerFaceIntegrator<2>(diff_stack, fec.get(), 1.0, num_state, alpha));
        }
    }

    template <int dim, bool entvar>
    void DGDSolver<dim, entvar>::addEntVolumeIntegrators()
    {
        ent->AddDomainIntegrator(new EntropyIntegrator<dim, entvar>(diff_stack));
    }

    template <int dim, bool entvar>
    void DGDSolver<dim, entvar>::initialHook(const ParCentGridFunction &state)
    {
        // res_norm0 is used to compute the time step in PTC
        res_norm0 = calcResidualNorm(state);
    }

    template <int dim, bool entvar>
    bool DGDSolver<dim, entvar>::iterationExit(int iter, double t, double t_final,
                                               double dt,
                                               const ParCentGridFunction &state)
    {
        // use tolerance options for Newton's method
        double norm = calcResidualNorm(state);
        if (norm <= 1e-11)
            return true;
        if (isnan(norm))
            return true;
    }

    template <int dim, bool entvar>
    void DGDSolver<dim, entvar>::addOutputs()
    {
        auto &fun = options["outputs"];
        int idx = 0;
        if (fun.find("drag") != fun.end())
        {
            // drag on the specified boundaries
            vector<int> tmp = fun["drag"].template get<vector<int>>();
            output_bndry_marker[idx].SetSize(tmp.size(), 0);
            output_bndry_marker[idx].Assign(tmp.data());
            output.emplace("drag", fes.get());
            mfem::Vector drag_dir(dim);
            drag_dir = 0.0;
            if (dim == 1)
            {
                drag_dir(0) = 1.0;
            }
            else
            {
                drag_dir(iroll) = cos(aoa_fs);
                drag_dir(ipitch) = sin(aoa_fs);
            }
            drag_dir *= 1.0 / pow(mach_fs, 2.0); // to get non-dimensional Cd
            output.at("drag").AddBdrFaceIntegrator(
                new DG_PressureForce<dim, entvar>(drag_dir, num_state, 1.0),
                output_bndry_marker[idx]);
            idx++;
        }
        if (fun.find("lift") != fun.end())
        {
            // lift on the specified boundaries
            vector<int> tmp = fun["lift"].template get<vector<int>>();
            output_bndry_marker[idx].SetSize(tmp.size(), 0);
            output_bndry_marker[idx].Assign(tmp.data());
            output.emplace("lift", fes.get());
            mfem::Vector lift_dir(dim);
            lift_dir = 0.0;
            if (dim == 1)
            {
                lift_dir(0) = 0.0;
            }
            else
            {
                lift_dir(iroll) = -sin(aoa_fs);
                lift_dir(ipitch) = cos(aoa_fs);
            }
            lift_dir *= 1.0 / pow(mach_fs, 2.0); // to get non-dimensional Cl
            output.at("lift").AddBdrFaceIntegrator(
                new DG_PressureForce<dim, entvar>(lift_dir, num_state, 1.0),
                output_bndry_marker[idx]);
            idx++;
        }
        if (fun.find("entropy") != fun.end())
        {
            // integral of entropy over the entire volume domain
            output.emplace("entropy", fes.get());
            output.at("entropy").AddDomainIntegrator(
                new EntropyIntegrator<dim, entvar>(diff_stack));
        }
    }

    template <int dim, bool entvar>
    double DGDSolver<dim, entvar>::calcStepSize(int iter, double t,
                                                double t_final,
                                                double dt_old,
                                                const ParCentGridFunction &state)
    {
        // ramp up time step for pseudo-transient continuation
        // TODO: the l2 norm of the weak residual is probably not ideal here
        // A better choice might be the l1 norm
        double res_norm = calcResidualNorm(state);
        double exponent = options["time-dis"]["res-exp"].template get<double>();
        double dt = options["time-dis"]["dt"].template get<double>();
        dt = dt * pow(res_norm0 / res_norm, exponent);
        return max(dt, dt_old);
    }

    template <int dim, bool entvar>
    void DGDSolver<dim, entvar>::getFreeStreamState(mfem::Vector &q_ref)
    {
        q_ref = 0.0;
        q_ref(0) = 1.0;
        if (dim == 1)
        {
            q_ref(1) = q_ref(0) * mach_fs; // ignore angle of attack
        }
        else
        {
            q_ref(iroll + 1) = q_ref(0) * mach_fs * cos(aoa_fs);
            q_ref(ipitch + 1) = q_ref(0) * mach_fs * sin(aoa_fs);
        }
        q_ref(dim + 1) = 1 / (euler::gamma * euler::gami) + 0.5 * mach_fs * mach_fs;
    }

    template <int dim, bool entvar>
    double DGDSolver<dim, entvar>::calcConservativeVarsL2Error(
        void (*u_exact)(const mfem::Vector &, mfem::Vector &), int entry)
    {
        // This lambda function computes the error at a node
        // Beware: this is not particularly efficient, given the conditionals
        // Also **NOT thread safe!**
        Vector qdiscrete(dim + 2), qexact(dim + 2); // define here to avoid reallocation
        auto node_error = [&](const Vector &discrete, const Vector &exact) -> double {
            if (entvar)
            {
                calcConservativeVars<double, dim>(discrete.GetData(),
                                                  qdiscrete.GetData());
                calcConservativeVars<double, dim>(exact.GetData(), qexact.GetData());
            }
            else
            {
                qdiscrete = discrete;
                qexact = exact;
            }
            double err = 0.0;
            if (entry < 0)
            {
                for (int i = 0; i < dim + 2; ++i)
                {
                    double dq = qdiscrete(i) - qexact(i);
                    err += dq * dq;
                }
            }
            else
            {
                err = qdiscrete(entry) - qexact(entry);
                err = err * err;
            }
            return err;
        };

        VectorFunctionCoefficient exsol(num_state, u_exact);
        DenseMatrix vals, exact_vals;
        Vector u_j, exsol_j;
        double loc_norm = 0.0;
        for (int i = 0; i < fes->GetNE(); i++)
        {
            const FiniteElement *fe = fes->GetFE(i);
            const IntegrationRule *ir;
            int intorder = 2 * fe->GetOrder() + 3;
            ir = &(IntRules.Get(fe->GetGeomType(), intorder));
            ElementTransformation *T = fes->GetElementTransformation(i);
            u->GetVectorValues(*T, *ir, vals);
            exsol.Eval(exact_vals, *T, *ir);
            for (int j = 0; j < ir->GetNPoints(); j++)
            {
                const IntegrationPoint &ip = ir->IntPoint(j);
                T->SetIntPoint(&ip);
                vals.GetColumnReference(j, u_j);
                exact_vals.GetColumnReference(j, exsol_j);
                loc_norm += ip.weight * T->Weight() * node_error(u_j, exsol_j);
            }
        }
        double norm;
        MPI_Allreduce(&loc_norm, &norm, 1, MPI_DOUBLE, MPI_SUM, comm);
        if (norm < 0.0) // This was copied from mfem...should not happen for us
        {
            return -sqrt(-norm);
        }
        return sqrt(norm);
    }

    template <int dim, bool entvar>
    void DGDSolver<dim, entvar>::convertToEntvar(mfem::Vector &state)
    {
        if (entvar)
        {
            return;
        }
        else
        {
            int num_nodes, offset;
            Array<int> vdofs(num_state);
            Vector el_con, el_ent;
            const FiniteElement *fe;
            for (int i = 0; i < fes->GetNE(); i++)
            {
                fe = fes->GetFE(i);
                num_nodes = fe->GetDof();
                for (int j = 0; j < num_nodes; j++)
                {
                    offset = i * num_nodes * num_state + j * num_state;
                    for (int k = 0; k < num_state; k++)
                    {
                        vdofs[k] = offset + k;
                    }
                    u->GetSubVector(vdofs, el_con);
                    calcEntropyVars<double, dim>(el_con.GetData(), el_ent.GetData());
                    state.SetSubVector(vdofs, el_ent);
                }
            }
        }
    }

    template <int dim, bool entvar>
    void DGDSolver<dim, entvar>::setSolutionError(
        void (*u_exact)(const mfem::Vector &, mfem::Vector &))
    {
        VectorFunctionCoefficient exsol(num_state, u_exact);
        GridFunType ue(fes.get());
        ue.ProjectCoefficient(exsol);
        // TODO: are true DOFs necessary here?
        HypreParVector *u_true = u->GetTrueDofs();
        HypreParVector *ue_true = ue.GetTrueDofs();
        *u_true -= *ue_true;
        u->SetFromTrueDofs(*u_true);
    }

    // explicit instantiation
    template class DGDSolver<1, true>;
    template class DGDSolver<1, false>;
    template class DGDSolver<2, true>;
    template class DGDSolver<2, false>;
    template class DGDSolver<3, true>;
    template class DGDSolver<3, false>;

} // namespace mach
