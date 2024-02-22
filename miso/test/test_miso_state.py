import unittest
import numpy as np
import openmdao.api as om
from openmdao.utils.assert_utils import assert_check_partials, assert_check_totals

from miso import PDESolver, MISOState, MISOMesh, MISOFunctional

class TestEMState(unittest.TestCase):
    em_options = {
        "mesh": {
            # "file": "data/testOMMISO/parallel_wires.smb",
            # "model-file": "data/testOMMISO/parallel_wires.egads",
            "file": "data/box.mesh",
            "refine": 0
        },
        "space-dis": {
            "basis-type": "nedelec",
            "degree": 1
        },
        "time-dis": {
            "steady": True,
        },
        "nonlin-solver": {
            "type": "newton",
            "printlevel": 2,
            "maxiter": 5,
            "reltol": 1e-6,
            "abstol": 1e-6
        },
        "lin-solver": {
            "type": "minres",
            "printlevel": 1,
            "maxiter": 100,
            "abstol": 1e-14,
            "reltol": 1e-14
        },
        "adj-solver": {
            "type": "minres",
            "printlevel": 1,
            "maxiter": 100,
            "abstol": 1e-14,
            "reltol": 1e-14
        },
        "lin-prec": {
            "printlevel": -1
        },
        "components": {
            # "wires": {
            #     "material": "copperwire",
            #     "attrs": [1, 2],
            #     "linear": True
            # },
            "attr1": {
                "attr": 1,
                "material": {
                    "name": "box1",
                    "mu_r": 795774.7154594767
                },
            },
            "attr2": {
                "attr": 2,
                "material": {
                    "name": "box1",
                    "mu_r": 795774.7154594767
                },
            }
        },
        "bcs": {
            "essential": "all"
        },
        "current": {
            "wires": {
                "z": [1, 2]
            }
        }
    }
    # def test_forward(self):
    #     prob = om.Problem()

    #     emSolver = PDESolver(type="magnetostatic", solver_options=em_options, comm=prob.comm)

    #     prob.model.add_subsystem("ivc",
    #                              MISOMesh(solver=emSolver),
    #                              promotes_outputs=["*"])
    #     prob.model.add_subsystem("em_solver",
    #                              MISOState(solver=emSolver, 
    #                                        depends=["current_density:wires"],
    #                                        check_partials=True),
    #                              promotes_inputs=["*"],
    #                              promotes_outputs=["state"])

    #     prob.set_solver_print(level=0)
    #     prob.setup()
    #     prob["current_density:wires"] = 3e6
    #     prob.run_model()

    def test_partials(self):
        prob = om.Problem()

        emSolver = PDESolver(type="magnetostatic", solver_options=self.em_options, comm=prob.comm)

        prob.model.add_subsystem("ivc",
                                 MISOMesh(solver=emSolver),
                                 promotes_outputs=["*"])
        solver = prob.model.add_subsystem("em_solver",
                                          MISOState(solver=emSolver, 
                                                    depends=["current_density:wires", "mesh_coords"],
                                                    check_partials=True),
                                          promotes_inputs=["current_density:wires", ("mesh_coords", "x_em0")],
                                          promotes_outputs=["state"])
        solver.set_check_partial_options(wrt="*",
                                         directional=False,
                                         form="central")

        prob.set_solver_print(level=0)
        prob.setup()
        # prob["current_density:wires"] = 1.0
        prob.run_model()

        data = prob.check_partials()
        # om.partial_deriv_plot("state", "state", data, jac_method="J_rev", binary = False)
        om.partial_deriv_plot("state", "mesh_coords", data, jac_method="J_rev", binary = False)
        # om.partial_deriv_plot("state", "current_density:wires", data,jac_method="J_rev", binary = False)
        assert_check_partials(data)

    def test_totals(self):
        prob = om.Problem()
        emSolver = PDESolver(type="magnetostatic", solver_options=self.em_options, comm=prob.comm)

        prob.model.add_subsystem("ivc",
                                 MISOMesh(solver=emSolver),
                                 promotes_outputs=["*"])
        prob.model.add_subsystem("em_solver",
                                 MISOState(solver=emSolver, 
                                           depends=["current_density:wires", "mesh_coords"],
                                           check_partials=True),
                                 promotes_inputs=["current_density:wires", ("mesh_coords", "x_em0")],
                                 promotes_outputs=["state"])

        prob.setup(mode="rev")
        # prob["current_density:wires"] = 1e6
        # om.n2(problem)
        prob.run_model()

        data = prob.check_totals(of=["state"], wrt=["current_density:wires"])
        assert_check_totals(data, atol=1e-6, rtol=1e-6)

class TestEMState2D(unittest.TestCase):
    square_options = {
        "mesh": {
            "file": "data/simple_square.mesh",
            "refine": 0
        },
        "space-dis": {
            "basis-type": "h1",
            "degree": 1
        },
        "lin-solver": {
            "type": "gmres",
            "printlevel": 1,
            "maxiter": 100,
            "abstol": 1e-14,
            "reltol": 1e-14
        },
        "lin-prec": {
            # "type": "hypreboomeramg"
            "type": "hypreilu"
        },
        "adj-solver": {
            "type": "gmres",
            "printlevel": 1,
            "maxiter": 100,
            "abstol": 1e-14,
            "reltol": 1e-14
        },
        "nonlin-solver": {
            "type": "newton",
            "printlevel": 1,
            "maxiter": 5,
            "reltol": 1e-12,
            "abstol": 1e-12
        },
        "components": {
            "ring": {
                "attrs": [1],
                "material": {
                    "name": "box1",
                    "mu_r": 795774.7154594767
                },
            }
        },
        # "bcs": {
        #     "essential": "all"
        # },
        "current": {
            "test": {
                "z": [1]
            }
        }
    }

    def test_partials(self):
        prob = om.Problem()

        emSolver = PDESolver(type="magnetostatic", solver_options=self.square_options, comm=prob.comm)

        prob.model.add_subsystem("ivc",
                                 MISOMesh(solver=emSolver),
                                 promotes_outputs=["*"])
        solver = prob.model.add_subsystem("em_solver",
                                          MISOState(solver=emSolver, 
                                                    depends=["current_density:test", "mesh_coords"],
                                                    check_partials=True),
                                          promotes_inputs=["current_density:test", ("mesh_coords", "x_em0")],
                                          promotes_outputs=["state"])
        solver.set_check_partial_options(wrt="*",
                                         directional=False,
                                         form="central",
                                         step=1e-5)

        prob.set_solver_print(level=0)
        prob.setup()

        state_size = emSolver.getFieldSize("state")
        prob["state"] = np.random.randn(state_size)

        data = prob.check_partials()
        # om.partial_deriv_plot("state", "state", data, jac_method="J_rev", binary = False)
        # om.partial_deriv_plot("state", "mesh_coords", data, jac_method="J_rev", binary = False)
        # om.partial_deriv_plot("state", "current_density:test", data,jac_method="J_rev", binary = False)
        assert_check_partials(data)

    def test_totals(self):
        prob = om.Problem()
        emSolver = PDESolver(type="magnetostatic", solver_options=self.square_options, comm=prob.comm)

        prob.model.add_subsystem("ivc",
                                 MISOMesh(solver=emSolver),
                                 promotes_outputs=["*"])
        prob.model.add_subsystem("em_solver",
                                 MISOState(solver=emSolver, 
                                           depends=["current_density:test", "mesh_coords"],
                                           check_partials=True),
                                 promotes_inputs=["current_density:test", ("mesh_coords", "x_em0")],
                                 promotes_outputs=["state"])

        prob.setup(mode="rev")
        state_size = emSolver.getFieldSize("state")
        state = np.random.randn(state_size)

        prob["state"] = state
        prob["current_density:test"] = 1.0
        prob.run_model()

        data = prob.check_totals(of=["state"],
                                 wrt=["current_density:test", "x_em0"],
                                 form="central",
                                 step=1e-5)
        assert_check_totals(data, atol=1e-6, rtol=1e-6)

class TestThermalState(unittest.TestCase):
    square_options = {
        "mesh": {
            "file": "data/simple_square.mesh",
            "refine": 0
        },
        "space-dis": {
            "basis-type": "h1",
            "degree": 1
        },
        "lin-solver": {
            "type": "pcg",
            "printlevel": 1,
            "maxiter": 100,
            "abstol": 1e-14,
            "reltol": 1e-14
        },
        "nonlin-solver": {
            "type": "newton",
            "printlevel": 1,
            "maxiter": 5,
            "reltol": 1e-12,
            "abstol": 1e-12
        },
        "components": {
            "test": {
                "attrs": [1],
                "material": {
                    "name": "test",
                    "kappa": 1.0,
                    "conductivity": {
                        "model": "linear",
                        "sigma_T_ref": 58.14e6,
                        "T_ref": 293.15,
                        "alpha_resistivity": 3.8e-3
                    }
                },
            }
        },
        "bcs": {
            "convection": [1, 2],
            "outflux": [3, 4]
        }
    }

    def test_partials(self):
        prob = om.Problem()

        thermal_solver = PDESolver(type="thermal",
                                   solver_options=self.square_options,
                                   comm=prob.comm)

        prob.model.add_subsystem("ivc",
                                 MISOMesh(solver=thermal_solver),
                                 promotes_outputs=["*"])

        solver = prob.model.add_subsystem("thermal_solver",
                                          MISOState(solver=thermal_solver, 
                                                    depends=["mesh_coords", "h", "fluid_temp", "thermal_load"],
                                                    check_partials=True),
                                          promotes_inputs=[("mesh_coords", "x_conduct0"), "h", "fluid_temp", "thermal_load"],
                                          promotes_outputs=["state"])
        solver.set_check_partial_options(wrt="*",
                                         directional=False,
                                         form="central",
                                         step=1e-5)

        prob.set_solver_print(level=0)
        prob.setup()

        state_size = thermal_solver.getFieldSize("state")
        prob["state"] = np.random.randn(state_size)
        prob["h"] = 1.0
        prob["fluid_temp"] = 1.0
        prob["thermal_load"] = np.random.randn(state_size)

        data = prob.check_partials()
        # om.partial_deriv_plot("state", "state", data, jac_method="J_rev", binary = False)
        # om.partial_deriv_plot("state", "mesh_coords", data, jac_method="J_rev", binary = False)
        # om.partial_deriv_plot("state", "current_density:test", data,jac_method="J_rev", binary = False)
        assert_check_partials(data)

    def test_totals(self):
        prob = om.Problem()
        thermal_solver = PDESolver(type="thermal", solver_options=self.square_options, comm=prob.comm)

        prob.model.add_subsystem("ivc",
                                 MISOMesh(solver=thermal_solver),
                                 promotes_outputs=["*"])
        prob.model.add_subsystem("thermal_solver",
                                 MISOState(solver=thermal_solver, 
                                           depends=["mesh_coords", "h", "fluid_temp", "thermal_load"],
                                           check_partials=True),
                                 promotes_inputs=[("mesh_coords", "x_conduct0"), "h", "fluid_temp", "thermal_load"],
                                 promotes_outputs=["state"])

        prob.setup(mode="rev")

        state_size = thermal_solver.getFieldSize("state")
        prob["state"] = np.random.randn(state_size)
        prob["h"] = 1.0
        prob["fluid_temp"] = 1.0
        prob["thermal_load"] = np.random.randn(state_size)
        prob.run_model()

        data = prob.check_totals(of=["state"],
                                 wrt=["x_conduct0", "h", "fluid_temp", "thermal_load"],
                                 form="central",
                                 step=1e-5)
        assert_check_totals(data, atol=1e-6, rtol=1e-6)

class TestElectroThermalState(unittest.TestCase):
    em_square_options = {
        "mesh": {
            "file": "data/simple_square.mesh",
            "refine": 0
        },
        "space-dis": {
            "basis-type": "h1",
            "degree": 1
        },
        "lin-solver": {
            "type": "gmres",
            "printlevel": 1,
            "maxiter": 100,
            "abstol": 1e-14,
            "reltol": 1e-14
        },
        "lin-prec": {
            # "type": "hypreboomeramg"
            "type": "hypreilu"
        },
        "adj-solver": {
            "type": "gmres",
            "printlevel": 1,
            "maxiter": 100,
            "abstol": 1e-14,
            "reltol": 1e-14
        },
        "nonlin-solver": {
            "type": "newton",
            "printlevel": 1,
            "maxiter": 5,
            "reltol": 1e-12,
            "abstol": 1e-12
        },
        "components": {
            "test": {
                "attrs": [1],
                "material": {
                    "name": "test",
                    # "mu_r": 795774.7154594767,
                    "reluctivity": {
                        "model": "lognu",
                        "cps": [5.5286, 5.4645, 4.5597, 4.2891, 3.8445, 4.2880, 4.9505, 11.9364, 11.9738, 12.6554, 12.8097, 13.3347, 13.5871, 13.5871, 13.5871],
                        "knots": [0, 0, 0, 0, 0.1479, 0.5757, 0.9924, 1.4090, 1.8257, 2.2424, 2.6590, 3.0757, 3.4924, 3.9114, 8.0039, 10.0000, 10.0000, 10.0000, 10.0000],
                        "degree": 3
                    },
                    "Demag": {
                        "T0": 293.15,
                        "alpha_B_r": -0.12,
                        "B_r_T0": 1.39,
                    },
                    "conductivity": {
                        "model": "linear",
                        "sigma_T_ref": 58.14e6,
                        "T_ref": 293.15,
                        "alpha_resistivity": 3.8e-3
                    },
                    "core_loss": {
                        "model": "CAL2",
                        "T0": 293.15,
                        "kh_T0": [5.97783049251564E-02, -6.58569751792524E-02, 3.52052785575931E-02, -6.54762513683037E-03],
                        "ke_T0": [3.83147202762929E-05, -4.19965038193089E-05, 2.09788988466414E-05, -3.88567697029196E-06],
                        "T1": 473.15,
                        "kh_T1": [5.78728253280150E-02, -7.94684973286488E-02, 5.09165213772802E-02, -1.11117379956941E-02],
                        "ke_T1": [3.20525407302126E-05, -1.43502199723297E-05, -3.74786590271071E-06, 2.68517704958978E-06]
                    },
                    "rho": 1.0
                },
            }
        },
        "bcs": {
            "essential": "all"
        },
        "current": {
            "test": {
                "z": [1]
            }
        },
        # "magnets": {
        #     "Nd2Fe14B" : {
        #         "north": [1]
        #     }
        # }
    }

    thermal_square_options = {
        "mesh": {
            "file": "data/simple_square.mesh",
            "refine": 0
        },
        "space-dis": {
            "basis-type": "h1",
            "degree": 1
        },
        "lin-solver": {
            "type": "pcg",
            "printlevel": 1,
            "maxiter": 100,
            "abstol": 1e-14,
            "reltol": 1e-14
        },
        "nonlin-solver": {
            "type": "newton",
            "printlevel": 1,
            "maxiter": 5,
            "reltol": 1e-12,
            "abstol": 1e-12
        },
        "components": {
            "test": {
                "attrs": [1],
                "material": {
                    "name": "test",
                    "kappa": 1.0,
                    "conductivity": {
                        "model": "linear",
                        "sigma_T_ref": 58.14e6,
                        "T_ref": 293.15,
                        "alpha_resistivity": 3.8e-3
                    }
                },
            }
        },
        "bcs": {
            "convection": [1, 2],
            "outflux": [3, 4]
        }
    }

    def test_partials(self):
        prob = om.Problem()

        em_solver = PDESolver(type="magnetostatic",
                              solver_options=self.em_square_options,
                              comm=prob.comm)


        thermal_solver = PDESolver(type="thermal",
                                   solver_options=self.thermal_square_options,
                                   comm=prob.comm)

        prob.model.add_subsystem("em_ivc",
                                 MISOMesh(solver=em_solver),
                                 promotes_outputs=["*"])

        prob.model.add_subsystem("therm_ivc",
                                 MISOMesh(solver=thermal_solver),
                                 promotes_outputs=["*"])

        em_solver_comp = prob.model.add_subsystem("em_solver",
                                             MISOState(solver=em_solver, 
                                                       depends=["current_density:test", "mesh_coords", "temperature"],
                                                       check_partials=True),
                                             promotes_inputs=["current_density:test", ("mesh_coords", "x_em0"), "temperature"],
                                             promotes_outputs=[("state", "em_state")])

        em_solver_comp.set_check_partial_options(wrt="*",
                                                directional=False,
                                                form="central",
                                                step=1e-5)
        
        flux_mag_comp = prob.model.add_subsystem("flux_magnitude",
                                                 MISOFunctional(solver=em_solver,
                                                                func="flux_magnitude",
                                                                depends=["state", "mesh_coords"],
                                                                check_partials=True),
                                                 promotes_inputs=[("state", "em_state"), ("mesh_coords", "x_em0")],
                                                 promotes_outputs=["flux_magnitude"])
        flux_mag_comp.set_check_partial_options(wrt="*",
                                                directional=False,
                                                form="central",
                                                step=1e-5)

        heat_source_inputs = [("mesh_coords", "x_em0"),
                               "temperature",
                               "frequency",
                               "wire_length",
                               "rms_current",
                               "strand_radius",
                               "strands_in_hand",
                               "stack_length",
                               ("peak_flux", "flux_magnitude"),
                               "model_depth",
                               "num_turns",
                               "num_slots"]
        heat_source = prob.model.add_subsystem("heat_source",
                                               MISOFunctional(solver=em_solver,
                                                              func="heat_source",
                                                              func_options={
                                                                    "dc_loss": {
                                                                        "attributes": [1]
                                                                    },
                                                                    "ac_loss": {
                                                                        "attributes": [1]
                                                                    },
                                                                    "core_loss": {
                                                                        "attributes": [1]
                                                                    },
                                                                },
                                                                depends=heat_source_inputs,
                                                                check_partials=True),
                                                promotes_inputs=heat_source_inputs,
                                                promotes_outputs=[("heat_source", "thermal_load")])
        heat_source.set_check_partial_options(wrt="*",
                                              directional=False,
                                              form="central",
                                              step=1e-5)

        heat_source.set_check_partial_options(wrt="rms_current",
                                              directional=False,
                                              form="central",
                                              step=1e-2)


        therm_solver_comp = prob.model.add_subsystem("thermal_solver",
                                                     MISOState(solver=thermal_solver, 
                                                               depends=["mesh_coords", "h", "fluid_temp", "thermal_load"],
                                                               check_partials=True),
                                                     promotes_inputs=[("mesh_coords", "x_conduct0"), "h", "fluid_temp", "thermal_load"],
                                                     promotes_outputs=[("state", "temperature")])

        therm_solver_comp.set_check_partial_options(wrt="*",
                                               directional=False,
                                               form="central",
                                               step=1e-5)

        prob.set_solver_print(level=0)
        prob.setup()

        state_size = em_solver.getFieldSize("state")
        prob["em_state"] = np.random.randn(state_size)
        # prob["temperature"][:] = 375

        prob["rms_current"] = 10

        state_size = thermal_solver.getFieldSize("state")
        prob["temperature"] = 30*np.random.randn(state_size) + 300
        prob["h"] = 1.0
        prob["fluid_temp"] = 1.0
        # prob["thermal_load"] = np.random.randn(state_size)

        data = prob.check_partials()
        # om.partial_deriv_plot("state", "state", data, jac_method="J_rev", binary = False)
        # om.partial_deriv_plot("state", "mesh_coords", data, jac_method="J_rev", binary = False)
        # om.partial_deriv_plot("state", "current_density:test", data,jac_method="J_rev", binary = False)
        assert_check_partials(data)

    def test_totals(self):
        prob = om.Problem()

        em_solver = PDESolver(type="magnetostatic",
                              solver_options=self.em_square_options,
                              comm=prob.comm)


        thermal_solver = PDESolver(type="thermal",
                                   solver_options=self.thermal_square_options,
                                   comm=prob.comm)

        prob.model.add_subsystem("em_ivc",
                                 MISOMesh(solver=em_solver),
                                 promotes_outputs=["*"])

        prob.model.add_subsystem("therm_ivc",
                                 MISOMesh(solver=thermal_solver),
                                 promotes_outputs=["*"])

        em_solver_comp = prob.model.add_subsystem("em_solver",
                                             MISOState(solver=em_solver, 
                                                       depends=["current_density:test", "mesh_coords", "temperature"],
                                                       check_partials=True),
                                             promotes_inputs=["current_density:test", ("mesh_coords", "x_em0"), "temperature"],
                                             promotes_outputs=[("state", "em_state")])

        em_solver_comp.set_check_partial_options(wrt="*",
                                                directional=False,
                                                form="central",
                                                step=1e-5)
        
        flux_mag_comp = prob.model.add_subsystem("flux_magnitude",
                                                 MISOFunctional(solver=em_solver,
                                                                func="flux_magnitude",
                                                                depends=["state", "mesh_coords"],
                                                                check_partials=True),
                                                 promotes_inputs=[("state", "em_state"), ("mesh_coords", "x_em0")],
                                                 promotes_outputs=["flux_magnitude"])
        flux_mag_comp.set_check_partial_options(wrt="*",
                                                directional=False,
                                                form="central",
                                                step=1e-5)

        heat_source_inputs = [("mesh_coords", "x_em0"),
                               "temperature",
                               "frequency",
                               "wire_length",
                               "rms_current",
                               "strand_radius",
                               "strands_in_hand",
                               "stack_length",
                               ("peak_flux", "flux_magnitude"),
                               "model_depth",
                               "num_turns",
                               "num_slots"]
        heat_source = prob.model.add_subsystem("heat_source",
                                               MISOFunctional(solver=em_solver,
                                                              func="heat_source",
                                                              func_options={
                                                                    "dc_loss": {
                                                                        "attributes": [1]
                                                                    },
                                                                    "ac_loss": {
                                                                        "attributes": [1]
                                                                    },
                                                                    "core_loss": {
                                                                        "attributes": [1]
                                                                    },
                                                                },
                                                                depends=heat_source_inputs,
                                                                check_partials=True),
                                                promotes_inputs=heat_source_inputs,
                                                promotes_outputs=[("heat_source", "thermal_load")])
        heat_source.set_check_partial_options(wrt="*",
                                              directional=False,
                                              form="central",
                                              step=1e-5)

        heat_source.set_check_partial_options(wrt="rms_current",
                                              directional=False,
                                              form="central",
                                              step=1e-2)


        therm_solver_comp = prob.model.add_subsystem("thermal_solver",
                                                     MISOState(solver=thermal_solver, 
                                                               depends=["mesh_coords", "h", "fluid_temp", "thermal_load"],
                                                               check_partials=True),
                                                     promotes_inputs=[("mesh_coords", "x_conduct0"), "h", "fluid_temp", "thermal_load"],
                                                     promotes_outputs=[("state", "temperature")])

        therm_solver_comp.set_check_partial_options(wrt="*",
                                               directional=False,
                                               form="central",
                                               step=1e-5)

        prob.setup(mode="rev")

        prob.model.nonlinear_solver = om.NonlinearBlockGS(maxiter=50, iprint=2,
                                                          atol=1e-6, rtol=1e-6,
                                                          use_aitken=False)
        
        # prob.model.linear_solver = om.DirectSolver(assemble_jac=False)
        prob.model.linear_solver = om.PETScKrylov(maxiter=25, iprint=2,
                                                  atol=1e-6, rtol=1e-6,
                                                  restart=25)
            
        prob.model.linear_solver.precon = om.LinearBlockGS(maxiter=1, iprint=2,
                                                           atol=1e-8, rtol=1e-8,
                                                           use_aitken=False)

        state_size = em_solver.getFieldSize("state")
        prob["em_state"] = np.random.randn(state_size)
        # prob["temperature"][:] = 375

        prob["rms_current"] = 10

        state_size = thermal_solver.getFieldSize("state")
        prob["temperature"] = 30*np.random.randn(state_size) + 300
        prob["h"] = 100.0
        prob["fluid_temp"] = 1.0
        # prob["thermal_load"] = np.random.randn(state_size)

        prob.run_model()

        data = prob.check_totals(of=["temperature"],
                                #  wrt=["x_conduct0", "h", "fluid_temp", "x_em0",],
                                 wrt=["h", "fluid_temp"],
                                 form="central",
                                 step=1e-5)
        assert_check_totals(data, atol=1e-6, rtol=1e-6)


if __name__ == "__main__":
    unittest.main()