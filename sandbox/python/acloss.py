import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from mach import MachSolver, Mesh, Vector


options = {
    "silent": False,
    "print-options": False,
    "mesh": {
        "file": "../../mach/test/data/testSolver/wire.smb",
        "model-file": "../../mach/test/data/testSolver/wire.egads"
    },
    "space-dis": {
        "basis-type": "nedelec",
        "degree": 2
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
        "current-density": 1.0,
        "current": {
            "z": [1]
        }
    }
}

def getK(N):
    if N == 1:
        return 1.0
    elif N == 3:
        return 1.55
    elif N == 9:
        return 1.84
    elif N == 27:
        return 1.92
    elif N > 27:
        return 2.0

def calcLitzResistance(n, r_s, r_0, f, sigma = 58.14e6, H = 0.0, n_b = 1, n_c = 1):
    K = getK(n)
    G = (3.7163312378020414 * r_s*2 * np.sqrt(f) ) ** 4 # r_s in meters
    Rac_Rdc = H + K * ((n * r_s / r_0) ** 2) * G

    R_dc_strand = 1.0 / (sigma * np.pi * r_s ** 2) # Ohm/meter
    R_dc = R_dc_strand * ((1.015) ** n_b) * ((1.025) ** n_c) / n
    R_ac = Rac_Rdc * R_dc
    return R_ac, R_dc

def calcLitzLoss(current_density, fill_factor, n, r_s, r_0, f, H = 0.0, n_b = 1, n_c = 1):
    R_ac, R_dc = calcLitzResistance(n, r_s, r_0, f, H=H, n_b=n_b, n_c=n_c)

    strand_area = np.pi * r_s ** 2
    litz_current = current_density * n*strand_area
    acloss = R_ac * litz_current ** 2
    dcloss = R_dc * litz_current ** 2
    return acloss, dcloss


def calcFEMLoss(solver, state, current_density, fill_factor, n, r_s, freq):
    inputs = {
        "current-density": current_density,
        "fill-factor": fill_factor,
        "state": state
    }
    solver.solveForState(inputs, state)

    inputs = {
        "diam": r_s*2,
        "frequency": freq,
        "fill-factor": fill_factor,
        "num-strands": float(n),
        "state": state
    }
    acloss = solver.calcOutput("ACLoss", inputs);

    inputs = {
        "fill-factor": fill_factor,
        "current-density": current_density,
        "state": state
    }
    dcloss = solver.calcOutput("DCLoss", inputs);

    return acloss, dcloss


if __name__ == "__main__":
    solver = MachSolver("Magnetostatic", options)
    solver.createOutput("ACLoss");
    solver.createOutput("DCLoss");

    state = solver.getNewField()
    zero = Vector(np.array([0.0, 0.0, 0.0]))
    solver.setFieldValue(state, zero);

    # 26 gauge wire for real motor
    d_0 = 0.002
    r_0 = d_0 / 2.0
    length = 0.001

    nsamples = 4
    freqs = np.logspace(2, 4, nsamples)
    fem_ac = np.zeros(nsamples)
    litz_ac = np.zeros(nsamples)
    litz_Rac = np.zeros(nsamples)

    for i in range(nsamples):
        # freq = 1e3 # 1000 Hz
        freq = float(freqs[i])
        print(freq)
        sigma = 58.14e6 # electrical conductivity of copper 
        current_density = 11e6 # 11 A/mm^2

        # # 3 strands in bundle
        # n = 3
        # r_s = float(1 / (1+2/np.sqrt(3)) * r_0)
        # filename = "acloss3strand.png"

        # # 27 strands
        # n = 27 
        # r_s = 0.169307931135 * r_0
        # filename = "acloss27strand.png"

        # 450 strands
        n = 450
        r_s = 0.043571578291 * r_0
        filename = "acloss450strand.png"

        strand_area = np.pi * r_s ** 2
        fill_factor = float(n*strand_area / (np.pi*r_0**2))
        
        R_ac, R_dc = calcLitzResistance(n, r_s, r_0, freq)
        litz_Rac[i] = R_ac

        litzacloss, litzdcloss = calcLitzLoss(current_density, fill_factor, n, r_s, r_0, freq, n_b = 0, n_c = 0)
        acloss, dcloss = calcFEMLoss(solver, state, current_density, fill_factor, n, r_s, freq)

        fem_ac[i] = acloss / length
        litz_ac[i] = litzacloss
        print()
        print("Total FEM copper loss: ", (acloss + dcloss) * 1.0 / length)
        print("Total Litz wire copper loss: ", litzacloss + litzdcloss)
        print("FEM AC loss: ", acloss * 1.0 / length)
        print("Litz wire AC loss: ", litzacloss)
        print("FEM DC loss: " , dcloss * 1.0 / length)
        print("Litz wire DC loss: ", litzdcloss)

    print(fem_ac)
    print(litz_ac)
    print(freqs)

    print(fem_ac / litz_ac)

    fig, ax = plt.subplots()
    ax.loglog(freqs, fem_ac, label="Hybrid-FEM")
    ax.loglog(freqs, litz_ac, label="Litz")

    ax.set(xlabel='frequency (Hz)', ylabel='AC Loss (W)')
    ax.grid()
    ax.legend()
    plt.ylim((1e-7, 1e0))

    fig.savefig(filename)
    # plt.show()

    # fig, ax = plt.subplots()
    # ax.loglog(freqs, litz_Rac)
    # ax.set(xlabel='frequency (Hz)', ylabel='AC Resistance (Ohm/m)')
    # ax.grid()
    # fig.savefig("resistance.png")
    # plt.show()

    #######################################################
    # This is the example from the New England Wire site
    #######################################################
    # n_b = 2
    # n_c = 1
    # r_s = 3.995e-5 # 40 AWG
    # r_0 = 0.0011938
    # n = 450
    # f = 100 * 1e3

    # R_ac, R_dc = calcLitzResistance(n, r_s, r_0, f, n_b=n_b, n_c=n_c, H=1.0, sigma=5.275528344750871e7)

    # print(R_dc * 304.8)
    # print(R_ac * 304.8)


