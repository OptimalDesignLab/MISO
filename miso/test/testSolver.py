import unittest
import numpy as np
import tempfile
import os

from miso import MISOSolver, Mesh, Vector

class SolverRegressionTests(unittest.TestCase):
    # def test_steady_vortex(self):

    #     def buildQuarterAnnulusMesh(degree, num_rad, num_ang, path):
    #         """Generate quarter annulus mesh 

    #         Generates a high-order quarter annulus mesh with 2 * `num_rad` x `num_ang`
    #         triangles, and saves it to a temporary file `tmp/qa.mesh`.

    #         Parameters
    #         ----------
    #         degree : int
    #             polynomial degree of the mapping
    #         num_rad : int
    #             number of nodes in the radial direction
    #         num_ang : int 
    #             number of nodes in the angular direction
    #         path : str
    #             the path to save the mesh file
    #         """
    #         def map(rt):
    #             xy = np.zeros(2)
    #             xy[0] = (rt[0] + 1.0)*np.cos(rt[1]) # need + 1.0 to shift r away from origin
    #             xy[1] = (rt[0] + 1.0)*np.sin(rt[1])
    #             return xy

    #         def apply_map(coords):
    #             num_nodes = coords.size
    #             for i in range(0, num_nodes, 2):
    #                 # print(np.array([coords[i], coords[i+1]]), end=' -> ')
    #                 xy = map(np.array([coords[i], coords[i+1]]))
    #                 coords[i] = xy[0]
    #                 coords[i+1] = xy[1]
    #                 # print(np.array([coords[i], coords[i+1]]))


    #         mesh = Mesh(num_rad, num_ang, 2.0, np.pi*0.5, degree)

    #         miso_nodes = Vector()
    #         mesh.getNodes(miso_nodes)
    #         nodes = np.array(miso_nodes, copy=False)
    #         apply_map(nodes)
    #         mesh.Print(path)

    #     # exact solution for conservative variables
    #     def qexact(x, q):
    #         # heat capcity ratio for air
    #         gamma = 1.4
    #         # ratio minus one
    #         gami = gamma - 1.0

    #         q.setSize(4)
    #         ri = 1.0
    #         Mai = 0.5 # 0.95 
    #         rhoi = 2.0
    #         prsi = 1.0/gamma
    #         rinv = ri/np.sqrt(x[0]*x[0] + x[1]*x[1])
    #         rho = rhoi * (1.0 + 0.5*gami*Mai*Mai*(1.0 - rinv*rinv)) ** (1.0/gami)
    #         Ma = np.sqrt((2.0/gami)*( ( (rhoi/rho) ** gami) * 
    #                         (1.0 + 0.5*gami*Mai*Mai) - 1.0 ) )

    #         if x[0] > 1e-15:
    #             theta = np.arctan(x[1]/x[0])
    #         else:
    #             theta = np.pi/2.0

    #         press = prsi* ((1.0 + 0.5*gami*Mai*Mai) / 
    #                             (1.0 + 0.5*gami*Ma*Ma)) ** (gamma/gami)

    #         a = np.sqrt(gamma*press/rho)

    #         q[0] = rho
    #         q[1] = rho*a*Ma*np.sin(theta)
    #         q[2] = -rho*a*Ma*np.cos(theta)
    #         q[3] = press/gami + 0.5*rho*a*a*Ma*Ma

    #     # Provide the options explicitly for regression tests
    #     options = {
    #         "mesh": {
    #         },
    #         "print-options": True,
    #         "flow-param": {
    #             "miso": 1.0,
    #             "aoa": 0.0
    #         },
    #         "space-dis": {
    #             "degree": 1,
    #             "lps-coeff": 1.0,
    #             "basis-type": "csbp"
    #         },
    #         "time-dis": {
    #             "steady": True,
    #             "steady-abstol": 1e-12,
    #             "steady-restol": 1e-10,
    #             "ode-solver": "PTC",
    #             "t-final": 100,
    #             "dt": 1e12,
    #             "cfl": 1.0,
    #             "res-exp": 2.0
    #         },
    #         "bcs": {
    #             "vortex": [1, 1, 1, 0],
    #             "slip-wall": [0, 0, 0, 1]
    #         },
    #         "nonlin-solver": {
    #             "printlevel": 1,
    #             "maxiter": 5,
    #             "reltol": 1e-12,
    #             "abstol": 1e-12
    #         },
    #         "petscsolver": {
    #             "ksptype": "gmres",
    #             "pctype": "lu",
    #             "abstol": 1e-15,
    #             "reltol": 1e-15,
    #             "maxiter": 100,
    #             "printlevel": 0
    #         },
    #         "lin-solver": {
    #             "printlevel": 0,
    #             "filllevel": 3,
    #             "maxiter": 100,
    #             "reltol": 1e-2,
    #             "abstol": 1e-12
    #         },
    #         "saveresults": False,
    #         "outputs":
    #         { 
    #             "drag": [0, 0, 0, 1]
    #         }
    #     }

    #     l2_errors = []
    #     # for use_entvar in [False, True]:
    #     for use_entvar in [False]:

    #         if use_entvar:
    #             target_error = [0.0690131081, 0.0224304871, 0.0107753424, 0.0064387612]
    #         else:
    #             target_error = [0.0700148195, 0.0260625842, 0.0129909277, 0.0079317615]
    #             target_drag = [-0.7351994763, -0.7173671079, -0.7152435959, -0.7146853812]
    #             # target_drag = [-0.7355357753, -0.717524391, -0.7152446356, -0.7146853447]

    #         tmp = tempfile.gettempdir()
    #         filepath = os.path.join(tmp, "qa")
    #         mesh_degree = options["space-dis"]["degree"] + 1;
    #         for nx in range(1, 5):
    #             buildQuarterAnnulusMesh(mesh_degree, nx, nx, filepath)
    #             options["mesh"]["file"] = filepath + ".mesh"

    #             # define the appropriate exact solution based on entvar
    #             if use_entvar:
    #                 # uexact = wexact
    #                 raise NotImplementedError
    #             else:
    #                 uexact = qexact

    #             solver = MISOSolver("Euler", options, entvar=use_entvar)

    #             state = solver.getNewField()

    #             solver.setInitialCondition(state, uexact)
    #             solver.solveForState(state);

    #             # residual = solver.getNewField()
    #             # solver.calcResidual(state, residual)

    #             # solver.printFields("steady_vortex", [state, residual], ["state", "residual"])

    #             l2_error = solver.calcL2Error(state, uexact, 0)
    #             drag = solver.calcFunctional(state, "drag")

    #             l2_errors.append(l2_error)
    #             self.assertLessEqual(l2_error, target_error[nx-1])
    #             # self.assertAlmostEqual(l2_error, target_error[nx-1])

    #             # self.assertAlmostEqual(drag, target_drag[nx-1])
        
    #     print(l2_errors)

    def test_miso_inputs(self):
        def buildMesh(nx, ny, nz, path):
            """Generate simple 3D box mesh

            Creates mesh for the parallelepiped [0,1]x[0,1]x[0,1], divided into
            6 x `nx` x `ny` x 'nz' tetrahedrons and saves it to a file
            specified by the path.

            Parameters
            ----------
            nx : int
                number of nodes in the x direction
            ny : int
                number of nodes in the y direction
            nz : int 
                number of nodes in the z direction
            path : str
                the path to save the mesh file
            """
            mesh = Mesh(nx, ny, nz, 1.0, 1.0, 1.0)
            mesh.Print(path)

        # Provide the options explicitly for regression tests
        options = {
            "mesh": {
            },
            "print-options": False,
            "space-dis": {
                "degree": 1,
                "basis-type": "H1"
            },
            "time-dis": {
                "steady": True,
                "steady-abstol": 1e-12,
                "steady-restol": 1e-10,
                "ode-solver": "PTC",
                "t-final": 100,
                "dt": 1e12,
                "cfl": 1.0,
                "res-exp": 2.0
            },
            "nonlin-solver": {
                "printlevel": 1,
                "maxiter": 5,
                "reltol": 1e-12,
                "abstol": 1e-12
            },
            "lin-solver": {
                "printlevel": 0,
                "filllevel": 3,
                "maxiter": 100,
                "reltol": 1e-2,
                "abstol": 1e-12
            },
            "external-fields": {
                "test_field": {
                    "basis-type": "H1",
                    "degree": 1,
                    "num-states": 1
                }
            }
        }

        tmp = tempfile.gettempdir()
        filepath = os.path.join(tmp, "qa")
        mesh_degree = options["space-dis"]["degree"] + 1;
        buildMesh(2, 2, 2, filepath)
        options["mesh"]["file"] = filepath + ".mesh"

        solver = MISOSolver("TestMISOInput", options)
        solver.createOutput("testMISOInput");

        state = solver.getNewField()
        solver.setFieldValue(state, 0.0);

        test_field = solver.getNewField();
        solver.setFieldValue(test_field, 0.0);

        inputs = {
            "test_val": 2.0,
            "test_field": test_field,
            "state": state
        }

        fun = solver.calcOutput("testMISOInput", inputs);
        self.assertAlmostEqual(fun, 2.0)

        inputs["test_val"] = 1.0;
        fun = solver.calcOutput("testMISOInput", inputs);
        self.assertAlmostEqual(fun, 1.0)

        inputs["test_val"] = 0.0;
        solver.setFieldValue(test_field, -1.0);
        fun = solver.calcOutput("testMISOInput", inputs);
        self.assertAlmostEqual(fun, -1.0)

    def test_ac_losses(self):
        # import Kelvin functions and derivatives
        from scipy.special import ber, bei, berp, beip

        # 26 gauge wire for real
        freq = 1e3
        mu_r = 1.0
        mu_0 = 4.0 * np.pi * 1e-7
        sigma = 58.14e6

        d_c = 0.002
        r = d_c / 2.0

        R_dc = 1.0 / (sigma * np.pi * r ** 2)
        print("R_dc: \t\t", R_dc)

        delta = 1 / np.sqrt(np.pi * freq * mu_r * mu_0 * sigma)
        print("delta: ", delta)

        q = d_c / (np.sqrt(2.0) * delta)
        R_ac = (np.sqrt(2.0) / (np.pi * d_c * sigma * delta)
                * (ber(q) * beip(q) - bei(q)*berp(q)) / (berp(q)**2 + beip(q)**2))
        print("R_ac: \t\t", R_ac)

        R_ac_approx = 1.0 / (np.pi * sigma * delta * 
                        (1 - np.exp(-r/delta)) * (2*r - delta*(1 - np.exp(-r/delta))))
        print("R_ac_approx: \t", R_ac_approx)

        K_s = 1.0
        x_s4 = ((8 * np.pi * freq * K_s) / (R_dc * 1e7))**2
        x_s = x_s4 ** 0.25
        # print("x_s: ", x_s)
        y_s = x_s4 / (192.0 + 0.8 * x_s4)
        # print("AC factor: ", 1.0 + y_s)
        R_ac_approx2 = R_dc * (1.0 + y_s)
        print("R_ac_approx2: \t", R_ac_approx2)
        print(2*"\n")

        area = np.pi * r ** 2
        current_density = 11e6
        current = current_density * area

        options = {
            "silent": False,
            "print-options": False,
            "mesh": {
                "file": "data/testSolver/wire.smb",
                "model-file": "data/testSolver/wire.egads"
            },
            "space-dis": {
                "basis-type": "nedelec",
                "degree": 1
            },
            "time-dis": {
                "steady": True,
                "steady-abstol": 1e-12,
                "steady-reltol": 1e-10,
                "ode-solver": "PTC",
                "t-final": 100,
                "dt": 1e12,
                "max-iter": 10
            },
            "lin-solver": {
                "type": "hypregmres",
                "printlevel": 0,
                "maxiter": 100,
                "abstol": 1e-14,
                "reltol": 1e-14
            },
            "lin-prec": {
                "type": "hypreams",
                "printlevel": 0
            },
            "nonlin-solver": {
                "type": "newton",
                "printlevel": 3,
                "maxiter": 5,
                "reltol": 1e-10,
                "abstol": 1e-8
            },
            "components": {
                "wire": {
                    "material": "copperwire",
                    "attr": 1,
                    "linear": True
                }
            },
            "bcs": {
                "essential": [1, 2, 3, 4]
            },
            "problem-opts": {
                "fill-factor": 1.0,
                "current_density": 1.0,
                "current": {
                    "z": [1]
                }
            }
        }

        solver = MISOSolver("Magnetostatic", options)
        solver.createOutput("ACLoss");
        solver.createOutput("DCLoss");

        state = solver.getNewField()
        zero = Vector(np.array([0.0, 0.0, 0.0]))
        solver.setFieldValue(state, zero);

        inputs = {
            "current_density": current_density,
            "fill-factor": 1.0,
            "state": state
        }
        solver.solveForState(inputs, state)

        inputs = {
            "diam": d_c,
            "frequency": freq,
            "fill-factor": 1.0,
            "state": state
        }
        acloss = solver.calcOutput("ACLoss", inputs);
        length = 0.001
        print("ACLoss val: ", acloss * 1.0 / length)
        print("Analtyical AC loss: ", R_ac * (current ** 2))
        # print("delta: ", delta)

        inputs = {
            "fill-factor": 1.0,
            "current_density": current_density,
            "state": state
        }
        dcloss = solver.calcOutput("DCLoss", inputs);
        length = 0.001
        print("DCLoss val: ", dcloss * 1.0 / length)
        print("Analtyical DC loss: ", R_dc * (current ** 2))

        print(2*"\n")
        print(20*"*")
        print("litz loss")
        print(20*"*")

        # 3 strands in bundle
        bundle_rad = r
        strand_rad = float(1 / (1+2/np.sqrt(3)) * bundle_rad)
        strand_area = np.pi * strand_rad ** 2
        n = 3
        strand_current = current_density * n*strand_area
        fill_factor = float(n*strand_area / (np.pi*bundle_rad**2))
        
        inputs = {
            "current_density": current_density,
            "fill-factor": fill_factor,
            "state": state
        }
        solver.solveForState(inputs, state)

        inputs = {
            "diam": strand_rad*2,
            "frequency": freq,
            "fill-factor": fill_factor,
            "state": state
        }
        acloss = solver.calcOutput("ACLoss", inputs);
        length = 0.001
        inputs = {
            "fill-factor": fill_factor,
            "current_density": current_density,
            "state": state
        }
        dcloss = solver.calcOutput("DCLoss", inputs);
        length = 0.001

        # print("strand_rad: ", strand_rad)
        X = 0.271*strand_rad*2*np.sqrt(freq*1e-6)
        H = 0.0
        K = 1.55
        G = (strand_rad*2 * np.sqrt(freq) / 10.44) ** 4
        Rac_Rdc = H + K * ((n * strand_rad / bundle_rad) ** 2) * G
        print("Rac / Rdc,", Rac_Rdc)
        R_dc_strand = 1.0 / (sigma * np.pi * strand_rad ** 2) # Ohm/meter

        print("strand rad: ", strand_rad)
        print("R_dc strand: ", R_dc_strand)
        n_b = 1
        n_c = 1
        R_dc = R_dc_strand * ((1.015) ** n_b) * ((1.025) ** n_c) / n
        # print("R_dc litz: ", R_dc)

        print("\n")

        print("FEM AC loss: ", acloss * 1.0 / length)
        # print("FEM DC loss: ", dcloss * 1.0 / length)

        # print("Litz wire DC loss: ", R_dc * strand_current ** 2)
        R_ac = Rac_Rdc * R_dc
        print("Litz wire AC loss: ", R_ac * strand_current ** 2)


        litz_loss = 1.0
        print(2*"\n")


if __name__ == '__main__':
    unittest.main()
