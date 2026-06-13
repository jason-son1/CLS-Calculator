def zigzag_chain():
    """1D zigzag chain model with flat band at E = -2."""
    return {
        "lattice": {
            "dimension": 1,
            "primitive_vectors": [[1.0]],
            "orbitals": [
                {"label": "A", "position": [0.0], "sublattice": 0},
                {"label": "B", "position": [0.5], "sublattice": 1}
            ]
        },
        "H_symbolic": [
            ["0", "sqrt(2) * (1 + exp(-I*kx))"],
            ["sqrt(2) * (1 + exp(I*kx))", "exp(I*kx) + exp(-I*kx)"]
        ],
        "options": {
            "k_grid": [100],
            "flat_tol": 1e-5
        }
    }

def kagome_nn():
    """Standard Kagome nearest-neighbor hopping model with flat band at E = -2 (t=1)."""
    return {
        "lattice": {
            "dimension": 2,
            "primitive_vectors": [[1.0, 0.0], [0.5, 0.8660254037844386]],
            "orbitals": [
                {"label": "A", "position": [0.5, 0.0], "sublattice": 0},
                {"label": "B", "position": [0.0, 0.5], "sublattice": 1},
                {"label": "C", "position": [0.5, 0.5], "sublattice": 2}
            ]
        },
        "H_symbolic": [
            ["0", "1 + exp(I*(kx - ky))", "1 + exp(-I*ky)"],
            ["1 + exp(-I*(kx - ky))", "0", "1 + exp(-I*kx)"],
            ["1 + exp(I*ky)", "1 + exp(I*kx)", "0"]
        ],
        "options": {
            "k_grid": [40, 40],
            "flat_tol": 1e-5
        }
    }

def bilayer_square():
    """Bilayer square lattice model with flat band at E = 2."""
    return {
        "lattice": {
            "dimension": 2,
            "primitive_vectors": [[1.0, 0.0], [0.0, 1.0]],
            "orbitals": [
                {"label": "A", "position": [0.0, 0.0], "sublattice": 0},
                {"label": "B", "position": [0.5, 0.5], "sublattice": 1}
            ]
        },
        "H_symbolic": [
            ["cos(kx) + cos(ky)", "cos(kx) + cos(ky) - 2"],
            ["cos(kx) + cos(ky) - 2", "cos(kx) + cos(ky)"]
        ],
        "options": {
            "k_grid": [40, 40],
            "flat_tol": 1e-5
        }
    }

def lieb():
    """Standard Lieb lattice model with flat band at E = 0."""
    return {
        "lattice": {
            "dimension": 2,
            "primitive_vectors": [[1.0, 0.0], [0.0, 1.0]],
            "orbitals": [
                {"label": "A", "position": [0.0, 0.0], "sublattice": 0},
                {"label": "B", "position": [0.5, 0.0], "sublattice": 1},
                {"label": "C", "position": [0.0, 0.5], "sublattice": 2}
            ]
        },
        "H_symbolic": [
            ["0", "1 + exp(-I*kx)", "0"],
            ["1 + exp(I*kx)", "0", "1 + exp(-I*ky)"],
            ["0", "1 + exp(I*ky)", "0"]
        ],
        "options": {
            "k_grid": [40, 40],
            "flat_tol": 1e-5
        }
    }

def modified_lieb():
    """Modified Lieb lattice model with flat band at E = 0."""
    return {
        "lattice": {
            "dimension": 2,
            "primitive_vectors": [[1.0, 0.0], [0.0, 1.0]],
            "orbitals": [
                {"label": "A", "position": [0.0, 0.0], "sublattice": 0},
                {"label": "B", "position": [0.5, 0.0], "sublattice": 1},
                {"label": "C", "position": [0.0, 0.5], "sublattice": 2}
            ]
        },
        "H_symbolic": [
            ["0", "1 - exp(I*kx)", "0"],
            ["1 - exp(-I*kx)", "0", "1 - exp(-I*ky)"],
            ["0", "1 - exp(I*ky)", "0"]
        ],
        "options": {
            "k_grid": [40, 40],
            "flat_tol": 1e-5
        }
    }

def checkerboard_1():
    """Checkerboard Model-I with flat band at E = 0, singular at (0,0)."""
    return {
        "lattice": {
            "dimension": 2,
            "primitive_vectors": [[1.0, 0.0], [0.0, 1.0]],
            "orbitals": [
                {"label": "A", "position": [0.0, 0.0], "sublattice": 0},
                {"label": "B", "position": [0.5, 0.5], "sublattice": 1}
            ]
        },
        "H_symbolic": [
            ["2 - 2*cos(ky)", "-(1 - exp(I*ky))*(1 - exp(-I*kx))"],
            ["-(1 - exp(-I*ky))*(1 - exp(I*kx))", "2 - 2*cos(kx)"]
        ],
        "options": {
            "k_grid": [40, 40],
            "flat_tol": 1e-5
        }
    }

def checkerboard_2():
    """Checkerboard Model-II with flat band at E = 0, singular at (pi, pi)."""
    return {
        "lattice": {
            "dimension": 2,
            "primitive_vectors": [[1.0, 0.0], [0.0, 1.0]],
            "orbitals": [
                {"label": "A", "position": [0.0, 0.0], "sublattice": 0},
                {"label": "B", "position": [0.5, 0.5], "sublattice": 1}
            ]
        },
        "H_symbolic": [
            ["2 + 2*cos(ky)", "-(1 + exp(I*ky))*(1 + exp(I*kx))"],
            ["-(1 + exp(-I*ky))*(1 + exp(-I*kx))", "2 + 2*cos(kx)"]
        ],
        "options": {
            "k_grid": [40, 40],
            "flat_tol": 1e-5
        }
    }

def checkerboard_3():
    """Checkerboard Model-III with isolated, non-singular flat band at E = 0."""
    return {
        "lattice": {
            "dimension": 2,
            "primitive_vectors": [[1.0, 0.0], [0.0, 1.0]],
            "orbitals": [
                {"label": "A", "position": [0.0, 0.0], "sublattice": 0},
                {"label": "B", "position": [0.5, 0.5], "sublattice": 1}
            ]
        },
        "H_symbolic": [
            ["5 + 4*cos(ky)", "-(1 + exp(-I*kx))*(2 + exp(I*ky))"],
            ["-(1 + exp(I*kx))*(2 + exp(-I*ky))", "2 + 2*cos(kx)"]
        ],
        "options": {
            "k_grid": [40, 40],
            "flat_tol": 1e-5
        }
    }

def honeycomb_flat():
    """Honeycomb flat model with flat band at E = 0."""
    return {
        "lattice": {
            "dimension": 2,
            "primitive_vectors": [[1.0, 0.0], [0.5, 0.8660254037844386]],
            "orbitals": [
                {"label": "A", "position": [0.0, 0.0], "sublattice": 0},
                {"label": "B", "position": [0.3333333333333333, 0.3333333333333333], "sublattice": 1}
            ]
        },
        "H_symbolic": [
            ["3 + exp(I*kx)*exp(-I*ky) + exp(-I*kx)*exp(I*ky) + exp(I*kx) + exp(-I*kx) + exp(I*ky) + exp(-I*ky)", "-1 - exp(-I*kx) - exp(-I*kx)*exp(I*ky)"],
            ["-1 - exp(I*kx) - exp(I*kx)*exp(-I*ky)", "1"]
        ],
        "options": {
            "k_grid": [40, 40],
            "flat_tol": 1e-5
        }
    }

def cubic_3D():
    """3D cubic lattice with three orbitals per site hosting flat band at E = 0."""
    return {
        "lattice": {
            "dimension": 3,
            "primitive_vectors": [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            "orbitals": [
                {"label": "px", "position": [0.0, 0.0, 0.0], "sublattice": 0},
                {"label": "py", "position": [0.0, 0.0, 0.0], "sublattice": 0},
                {"label": "pz", "position": [0.0, 0.0, 0.0], "sublattice": 0}
            ]
        },
        "H_symbolic": [
            ["0", "-I*sin(kz)", "sin(ky)"],
            ["I*sin(kz)", "0", "-sin(kx)"],
            ["sin(ky)", "-sin(kx)", "0"]
        ],
        "options": {
            "k_grid": [15, 15, 15],
            "flat_tol": 1e-5
        }
    }

def kagome_3():
    """Kagome-3 model hosting doubly degenerate non-singular flat bands at E = -2 (t=1)."""
    return {
        "lattice": {
            "dimension": 2,
            "primitive_vectors": [[1.0, 0.0], [0.5, 0.8660254037844386]],
            "orbitals": [
                {"label": "A", "position": [0.5, 0.0], "sublattice": 0},
                {"label": "B", "position": [0.0, 0.5], "sublattice": 1},
                {"label": "C", "position": [0.5, 0.5], "sublattice": 2}
            ]
        },
        "H_symbolic": [
            ["exp(I*kx) + exp(-I*kx)", "1 + exp(I*kx) + exp(I*ky) + exp(I*kx)*exp(I*ky)", "1 + exp(-I*kx) + exp(I*ky) + exp(I*kx)*exp(I*ky)"],
            ["1 + exp(-I*kx) + exp(-I*ky) + exp(-I*kx)*exp(-I*ky)", "exp(I*ky) + exp(-I*ky)", "1 + exp(-I*kx) + exp(I*ky) + exp(-I*kx)*exp(-I*ky)"],
            ["1 + exp(I*kx) + exp(-I*ky) + exp(-I*kx)*exp(-I*ky)", "1 + exp(I*kx) + exp(-I*ky) + exp(I*kx)*exp(I*ky)", "exp(I*kx)*exp(I*ky) + exp(-I*kx)*exp(-I*ky)"]
        ],
        "options": {
            "k_grid": [25, 25],
            "flat_tol": 1e-5
        }
    }


def flatband_5_trig():
    """Five-band model (Q=5) with a hub trio (E=0) coupled to a constant rim pair (E=-1); flat band at E=0 (M=1). User-provided test model 1: the CLS is the structural Schur reduction null(B^dagger) on the hub."""
    return {
        "lattice": {
            "dimension": 2,
            "primitive_vectors": [[1.0, 0.0], [0.0, 1.0]],
            "orbitals": [
                {"label": "o0", "position": [0.0, 0.0], "sublattice": 0},
                {"label": "o1", "position": [0.4, 0.0], "sublattice": 0},
                {"label": "o2", "position": [0.0, 0.4], "sublattice": 0},
                {"label": "o3", "position": [0.2, 0.3], "sublattice": 1},
                {"label": "o4", "position": [0.2, -0.3], "sublattice": 1}
            ]
        },
        "H_symbolic": [
            ['0', '0', '0', '2*sin(kx/2 - ky/2) + 2*I*sin(kx/2 + ky/2)', '2*cos(kx/2 - ky/2) + 2*cos(kx/2 + ky/2)'],
            ['0', '0', '0', '2*cos(ky/2)', '2*I*sin(ky/2)'],
            ['0', '0', '0', '-2*I*cos(kx/2)', '2*I*sin(kx/2)'],
            ['2*sin(kx/2 - ky/2) - 2*I*sin(kx/2 + ky/2)', '2*cos(ky/2)', '2*I*cos(kx/2)', '-1', '0'],
            ['2*cos(kx/2 - ky/2) + 2*cos(kx/2 + ky/2)', '-2*I*sin(ky/2)', '-2*I*sin(kx/2)', '0', '-1']
        ],
        "options": {
            "k_grid": [24, 24],
            "flat_tol": 1e-4
        }
    }


def flatband_5_sqrt3():
    """Five-band model (Q=5) with sqrt(3) (triangular) Bloch phases: hub trio (E=0) coupled to a constant rim pair (E=-1); flat band at E=0 (M=1). User-provided test model 2; exercises the real-axis GCD for irrational exponents."""
    return {
        "lattice": {
            "dimension": 2,
            "primitive_vectors": [[1.0, 0.0], [0.0, 1.0]],
            "orbitals": [
                {"label": "o0", "position": [0.0, 0.0], "sublattice": 0},
                {"label": "o1", "position": [0.4, 0.0], "sublattice": 0},
                {"label": "o2", "position": [0.0, 0.4], "sublattice": 0},
                {"label": "o3", "position": [0.2, 0.3], "sublattice": 1},
                {"label": "o4", "position": [0.2, -0.3], "sublattice": 1}
            ]
        },
        "H_symbolic": [
            ['0', '0', '0', 'exp(-2*I*pi/3)*exp(I*(-kx/2 + sqrt(3)*ky/6)) + exp(I*(kx/2 + sqrt(3)*ky/6)) + exp(2*I*pi/3)*exp(-sqrt(3)*I*ky/3)', 'exp(I*(-kx/2 - sqrt(3)*ky/6)) + exp(-2*I*pi/3)*exp(I*(kx/2 - sqrt(3)*ky/6)) + exp(2*I*pi/3)*exp(sqrt(3)*I*ky/3)'],
            ['0', '0', '0', 'exp(I*(-kx/2 + sqrt(3)*ky/6)) + exp(I*(kx/2 + sqrt(3)*ky/6)) + exp(-sqrt(3)*I*ky/3)', 'exp(I*(-kx/2 - sqrt(3)*ky/6)) + exp(I*(kx/2 - sqrt(3)*ky/6)) + exp(sqrt(3)*I*ky/3)'],
            ['0', '0', '0', 'exp(I*(-kx/2 + sqrt(3)*ky/6)) + exp(I*(kx/2 + sqrt(3)*ky/6)) + exp(-sqrt(3)*I*ky/3)', 'exp(sqrt(3)*I*ky/3) - exp(-I*(kx/2 + sqrt(3)*ky/6)) + exp(-I*(-kx/2 + sqrt(3)*ky/6))'],
            ['exp(-2*I*pi/3)*exp(sqrt(3)*I*ky/3) + exp(-I*(kx/2 + sqrt(3)*ky/6)) + exp(2*I*pi/3)*exp(-I*(-kx/2 + sqrt(3)*ky/6))', 'exp(sqrt(3)*I*ky/3) + exp(-I*(kx/2 + sqrt(3)*ky/6)) + exp(-I*(-kx/2 + sqrt(3)*ky/6))', 'exp(sqrt(3)*I*ky/3) + exp(-I*(kx/2 + sqrt(3)*ky/6)) + exp(-I*(-kx/2 + sqrt(3)*ky/6))', '-1', '0'],
            ['exp(-2*I*pi/3)*exp(-sqrt(3)*I*ky/3) + exp(2*I*pi/3)*exp(-I*(kx/2 - sqrt(3)*ky/6)) + exp(-I*(-kx/2 - sqrt(3)*ky/6))', 'exp(-sqrt(3)*I*ky/3) + exp(-I*(kx/2 - sqrt(3)*ky/6)) + exp(-I*(-kx/2 - sqrt(3)*ky/6))', 'exp(I*(-kx/2 + sqrt(3)*ky/6)) - exp(I*(kx/2 + sqrt(3)*ky/6)) + exp(-sqrt(3)*I*ky/3)', '0', '-1']
        ],
        "options": {
            "k_grid": [24, 24],
            "flat_tol": 1e-4
        }
    }


def flatband_10_sqrt3_deg():
    """Ten-band model (Q=10) with sqrt(3) phases: a six-orbital hub (E=0) coupled to a constant four-orbital rim (E=-1); doubly degenerate flat band at E=0 (M=2). User-provided test model 3 (hub on-site t=0). The 8x8 Faddeev-LeVerrier adjugate is float-unstable here, so the CLS is found via the structural Schur reduction (a stable 4x4 block)."""
    return {
        "lattice": {
            "dimension": 2,
            "primitive_vectors": [[1.0, 0.0], [0.0, 1.0]],
            "orbitals": [
                {"label": "o0", "position": [0.0, 0.0], "sublattice": 0},
                {"label": "o1", "position": [0.3, 0.0], "sublattice": 0},
                {"label": "o2", "position": [-0.15, 0.26], "sublattice": 0},
                {"label": "o3", "position": [-0.15, -0.26], "sublattice": 0},
                {"label": "o4", "position": [0.15, 0.0], "sublattice": 0},
                {"label": "o5", "position": [-0.075, 0.13], "sublattice": 0},
                {"label": "o6", "position": [0.2, 0.2], "sublattice": 1},
                {"label": "o7", "position": [-0.2, 0.2], "sublattice": 1},
                {"label": "o8", "position": [-0.2, -0.2], "sublattice": 1},
                {"label": "o9", "position": [0.2, -0.2], "sublattice": 1}
            ]
        },
        "H_symbolic": [
            ['0', '0', '0', '0', '0', '0', '-I*sin(kx/2)', 'I*sin(kx/2)', 'cos(kx/2)', '-cos(kx/2)'],
            ['0', '0', '0', '0', '0', '0', '-I*sin(kx/2)', '-I*sin(kx/2)', 'cos(kx/2)', 'cos(kx/2)'],
            ['0', '0', '0', '0', '0', '0', 'I*sin(kx/4 - sqrt(3)*ky/4)', 'I*(1/2 + sqrt(3)*I/2)*sin(kx/4 - sqrt(3)*ky/4)', '(-1/2 + sqrt(3)*I/2)*cos(kx/4 - sqrt(3)*ky/4)', '(1/2 - sqrt(3)*I/2)*cos(kx/4 - sqrt(3)*ky/4)'],
            ['0', '0', '0', '0', '0', '0', '-I*(1/2 - sqrt(3)*I/2)*sin(kx/4 - sqrt(3)*ky/4)', 'I*sin(kx/4 - sqrt(3)*ky/4)', '(-1/2 - sqrt(3)*I/2)*cos(kx/4 - sqrt(3)*ky/4)', '(-1/2 - sqrt(3)*I/2)*cos(kx/4 - sqrt(3)*ky/4)'],
            ['0', '0', '0', '0', '0', '0', 'I*sin(kx/4 + sqrt(3)*ky/4)', 'I*(1/2 - sqrt(3)*I/2)*sin(kx/4 + sqrt(3)*ky/4)', '(-1/2 - sqrt(3)*I/2)*cos(kx/4 + sqrt(3)*ky/4)', '(1/2 + sqrt(3)*I/2)*cos(kx/4 + sqrt(3)*ky/4)'],
            ['0', '0', '0', '0', '0', '0', '-I*(1/2 + sqrt(3)*I/2)*sin(kx/4 + sqrt(3)*ky/4)', 'I*sin(kx/4 + sqrt(3)*ky/4)', '(-1/2 + sqrt(3)*I/2)*cos(kx/4 + sqrt(3)*ky/4)', '(-1/2 + sqrt(3)*I/2)*cos(kx/4 + sqrt(3)*ky/4)'],
            ['I*sin(kx/2)', 'I*sin(kx/2)', '-I*sin(kx/4 - sqrt(3)*ky/4)', 'I*(1/2 + sqrt(3)*I/2)*sin(kx/4 - sqrt(3)*ky/4)', '-I*sin(kx/4 + sqrt(3)*ky/4)', 'I*(1/2 - sqrt(3)*I/2)*sin(kx/4 + sqrt(3)*ky/4)', '-1', '0', '0', '0'],
            ['-I*sin(kx/2)', 'I*sin(kx/2)', '-I*(1/2 - sqrt(3)*I/2)*sin(kx/4 - sqrt(3)*ky/4)', '-I*sin(kx/4 - sqrt(3)*ky/4)', '-I*(1/2 + sqrt(3)*I/2)*sin(kx/4 + sqrt(3)*ky/4)', '-I*sin(kx/4 + sqrt(3)*ky/4)', '0', '-1', '0', '0'],
            ['cos(kx/2)', 'cos(kx/2)', '(-1/2 - sqrt(3)*I/2)*cos(kx/4 - sqrt(3)*ky/4)', '(-1/2 + sqrt(3)*I/2)*cos(kx/4 - sqrt(3)*ky/4)', '(-1/2 + sqrt(3)*I/2)*cos(kx/4 + sqrt(3)*ky/4)', '(-1/2 - sqrt(3)*I/2)*cos(kx/4 + sqrt(3)*ky/4)', '0', '0', '-1', '0'],
            ['-cos(kx/2)', 'cos(kx/2)', '(1/2 + sqrt(3)*I/2)*cos(kx/4 - sqrt(3)*ky/4)', '(-1/2 + sqrt(3)*I/2)*cos(kx/4 - sqrt(3)*ky/4)', '(1/2 - sqrt(3)*I/2)*cos(kx/4 + sqrt(3)*ky/4)', '(-1/2 - sqrt(3)*I/2)*cos(kx/4 + sqrt(3)*ky/4)', '0', '0', '0', '-1']
        ],
        "options": {
            "k_grid": [25, 25],
            "flat_tol": 1e-4
        }
    }


def chern_3_flatband(t1=1.0, t2=1.0, t3=1.0, t4=1.0, t1_prime=0.5j, t2_prime=0.5j, t3_prime=0.5j, t4_prime=0.6j, delta=5.0):
    """Chern number 3 (C=3) flat-band model on bipartite skeleton (7-band model).
    L: 1a Wyckoff (triangular lattice), 4 orbitals = {s1, s2, f, p+}
    L~: 3c Wyckoff (kagome lattice), 3 orbitals = {p+ on each of the 3 sites}
    """
    import sympy
    kx, ky = sympy.symbols('kx ky', real=True)
    
    # Convert input parameters to SymPy
    t1_s = sympy.sympify(t1)
    t2_s = sympy.sympify(t2)
    t3_s = sympy.sympify(t3)
    t4_s = sympy.sympify(t4)
    t1_p = sympy.sympify(t1_prime)
    t2_p = sympy.sympify(t2_prime)
    t3_p = sympy.sympify(t3_prime)
    t4_p = sympy.sympify(t4_prime)
    delta_s = sympy.sympify(delta)
    
    # tau_j and tau_j' phase factors for j = 1, 2, 3
    tau1 = 0.5 * kx
    tau1_p = 0.5 * kx + ky
    tau2 = 0.5 * ky
    tau2_p = -kx - 0.5 * ky
    tau3 = -0.5 * (kx + ky)
    tau3_p = 0.5 * kx - 0.5 * ky

    # S(k) elements for j = 1, 2, 3
    S_01 = 2 * sympy.I * (t1_s * sympy.sin(tau1) + t1_p * sympy.sin(tau1_p))
    S_02 = 2 * sympy.I * (t1_s * sympy.sin(tau2) + t1_p * sympy.sin(tau2_p))
    S_03 = 2 * sympy.I * (t1_s * sympy.sin(tau3) + t1_p * sympy.sin(tau3_p))
    
    S_11 = 2 * sympy.I * (t2_s * sympy.sin(tau1) + t2_p * sympy.sin(tau1_p))
    S_12 = 2 * sympy.I * (t2_s * sympy.sin(tau2) + t2_p * sympy.sin(tau2_p))
    S_13 = 2 * sympy.I * (t2_s * sympy.sin(tau3) + t2_p * sympy.sin(tau3_p))
    
    S_21 = 2 * (t3_s * sympy.cos(tau1) + t3_p * sympy.cos(tau1_p))
    S_22 = -2 * (t3_s * sympy.cos(tau2) + t3_p * sympy.cos(tau2_p))
    S_23 = 2 * (t3_s * sympy.cos(tau3) + t3_p * sympy.cos(tau3_p))
    
    S_31 = 2 * (t4_s * sympy.cos(tau1) + t4_p * sympy.cos(tau1_p))
    S_32 = 2 * sympy.exp(-sympy.I * sympy.pi / 3) * (t4_s * sympy.cos(tau2) + t4_p * sympy.cos(tau2_p))
    S_33 = 2 * sympy.exp(-sympy.I * 2 * sympy.pi / 3) * (t4_s * sympy.cos(tau3) + t4_p * sympy.cos(tau3_p))
    
    S = sympy.Matrix([
        [S_01, S_02, S_03],
        [S_11, S_12, S_13],
        [S_21, S_22, S_23],
        [S_31, S_32, S_33]
    ])
    
    # Construct 7x7 Hamiltonian
    H = sympy.zeros(7, 7)
    H[0:4, 4:7] = S
    H[4:7, 0:4] = S.H # H is conjugate transpose
    
    for i in range(3):
        H[i + 4, i + 4] = -delta_s
        
    H_str = [[str(H[r, c]) for c in range(7)] for r in range(7)]
    
    return {
        "lattice": {
            "dimension": 2,
            "primitive_vectors": [[1.0, 0.0], [-0.5, 0.8660254037844386]],
            "orbitals": [
                {"label": "L_s1", "position": [0.0, 0.0], "sublattice": 0},
                {"label": "L_s2", "position": [0.0, 0.0], "sublattice": 0},
                {"label": "L_f", "position": [0.0, 0.0], "sublattice": 0},
                {"label": "L_p", "position": [0.0, 0.0], "sublattice": 0},
                {"label": "Ltilde_1", "position": [0.5, 0.0], "sublattice": 1},
                {"label": "Ltilde_2", "position": [0.0, 0.5], "sublattice": 1},
                {"label": "Ltilde_3", "position": [0.5, 0.5], "sublattice": 1}
            ]
        },
        "H_symbolic": H_str,
        "options": {
            "k_grid": [24, 24],
            "flat_tol": 1e-4,
            "k_path_str": "Γ - K - M - Γ"
        }
    }

