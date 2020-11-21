import unittest
import numpy as np
import tempfile
import os

from mach import omMach, Mesh, Vector, MachSolver
import openmdao.api as om

options = {
    'mesh': {
    },
    'print-options': False,
    'flow-param': {
        'mach': 1.0,
        'aoa': 0.0
    },
    'space-dis': {
        'degree': 1,
        'lps-coeff': 1.0,
        'basis-type': 'csbp'
    },
    'time-dis': {
        'steady': True,
        'steady-abstol': 1e-12,
        'steady-restol': 1e-10,
        'ode-solver': 'PTC',
        't-final': 100,
        'dt': 1e12,
        'cfl': 1.0,
        'res-exp': 2.0
    },
    'bcs': {
        'vortex': [1, 1, 1, 0],
        'slip-wall': [0, 0, 0, 1]
    },
    'newton': {
        'printlevel': 0,
        'maxiter': 50,
        'reltol': 1e-1,
        'abstol': 1e-12
    },
    'petscsolver': {
        'ksptype': 'gmres',
        'pctype': 'lu',
        'abstol': 1e-15,
        'reltol': 1e-15,
        'maxiter': 100,
        'printlevel': 0
    },
    'lin-solver': {
        'printlevel': 0,
        'filllevel': 3,
        'maxiter': 100,
        'reltol': 1e-2,
        'abstol': 1e-12
    },
    'saveresults': False,
    'outputs':
    { 
        'drag': [0, 0, 0, 1]
    }
}

def buildQuarterAnnulusMesh(degree, num_rad, num_ang, path):
    '''Generate quarter annulus mesh 

    Generates a high-order quarter annulus mesh with 2 * `num_rad` x `num_ang`
    triangles, and saves it to a temporary file `tmp/qa.mesh`.

    Parameters
    ----------
    degree : int
        polynomial degree of the mapping
    num_rad : int
        number of nodes in the radial direction
    num_ang : int 
        number of nodes in the angular direction
    path : str
        the path to save the mesh file
    '''
    def map(rt):
        xy = np.zeros(2)
        xy[0] = (rt[0] + 1.0)*np.cos(rt[1]) # need + 1.0 to shift r away from origin
        xy[1] = (rt[0] + 1.0)*np.sin(rt[1])
        return xy

    def apply_map(coords):
        num_nodes = coords.size
        for i in range(0, num_nodes, 2):
        # for i in range(0, num_nodes//2):
            # print(np.array([coords[i], coords[i+1]]), end=' -> ')
            # xy = map(np.array([coords[i], coords[i+num_nodes//2]]))
            xy = map(np.array([coords[i], coords[i+1]]))
            coords[i] = xy[0]
            coords[i+1] = xy[1]
            # print(np.array([coords[i], coords[i+1]]))


    mesh = Mesh(num_rad, num_ang, 2.0, np.pi*0.5, degree)

    mach_nodes = Vector()
    mesh.getNodes(mach_nodes)
    print(mach_nodes)
    nodes = np.array(mach_nodes, copy=False)
    apply_map(nodes)
    mesh.Print(path)
    mesh.PrintVTU("testpath")
    return mesh

# exact solution for conservative variables
def qexact(x, q):
    # heat capcity ratio for air
    gamma = 1.4
    # ratio minus one
    gami = gamma - 1.0

    q.setSize(4)
    ri = 1.0
    Mai = 0.5 # 0.95 
    rhoi = 2.0
    prsi = 1.0/gamma
    rinv = ri/np.sqrt(x[0]*x[0] + x[1]*x[1])
    rho = rhoi * (1.0 + 0.5*gami*Mai*Mai*(1.0 - rinv*rinv)) ** (1.0/gami)
    Ma = np.sqrt((2.0/gami)*( ( (rhoi/rho) ** gami) * 
                    (1.0 + 0.5*gami*Mai*Mai) - 1.0 ) )

    if x[0] > 1e-15:
        theta = np.arctan(x[1]/x[0])
    else:
        theta = np.pi/2.0

    press = prsi* ((1.0 + 0.5*gami*Mai*Mai) / 
                        (1.0 + 0.5*gami*Ma*Ma)) ** (gamma/gami)

    a = np.sqrt(gamma*press/rho)

    q[0] = rho
    q[1] = rho*a*Ma*np.sin(theta)
    q[2] = -rho*a*Ma*np.cos(theta)
    q[3] = press/gami + 0.5*rho*a*a*Ma*Ma

class TestMachGroup(unittest.TestCase):
    def test_group(self):
        
        # for use_entvar in [False, True]:
        for use_entvar in [False]:

            if use_entvar:
                pass
            else:
                # c++:
                # target_drag = [-0.7355357753, -0.717524391, -0.7152446356, -0.7146853447]
                # python:
                target_drag = [-0.7351994763, -0.7173671079, -0.7152435959, -0.7146853812]
                # openmdao:
                # target_drag = [-0.7817978779, -0.7323840978, -0.7176052052, -0.7146853447]
                # pass

            tmp = tempfile.gettempdir()
            filepath = os.path.join(tmp, "qa")
            mesh_degree = options["space-dis"]["degree"] + 1;
            for nx in range(1, 5):

                problem = om.Problem()

                model = problem.model

                ivc = model.add_subsystem('indeps', om.IndepVarComp(), promotes=['*'])

                ### build mesh and get nodes as input variable for the group
                options['mesh']['file'] = filepath + '.mesh'

                mesh = buildQuarterAnnulusMesh(mesh_degree, nx, nx, filepath)

                # mach_nodes = Vector()
                # mesh.getNodes(mach_nodes)
                # mesh_nodes = np.array(mach_nodes, copy=False)

                solver = MachSolver('Euler', options, problem.comm)
                mach_nodes = solver.getMeshCoordinates()
                mesh_nodes = np.array(mach_nodes, copy=False)

                ivc.add_output('vol_mesh_coords', mesh_nodes, mesh_nodes.shape)

                model.add_subsystem('machSolver', omMach(options_dict=options,
                                                            solver_type='Euler',
                                                            initial_condition=qexact),
                                    promotes_inputs=['vol_mesh_coords'],
                                    promotes_outputs=['func'])

                problem.setup()
                problem.run_model()
                drag = problem.get_val('func')
                print(drag[0])
                self.assertAlmostEqual(drag[0], target_drag[nx-1])

if __name__ == '__main__':
    unittest.main()