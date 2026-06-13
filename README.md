# CLS Finder - Compact Localization State & Topology Calculator

A Python-based toolkit for tight-binding models that automatically detects flat bands, derives their Compact Localization States (CLS) using both analytical (Minor + Adjugate) and numerical (IDFT on BZ grid) paths, classifies their topology (singular/non-singular), and generates real-space lattice plots and summary reports.

This program is built on the theoretical framework presented in:
> Jun-Won Rhim & Bohm-Jung Yang, *"Classification of flat bands according to the band-crossing singularity of Bloch wave functions"* (Phys. Rev. B, arXiv:1808.05926v2).

---

## 1. Features
- **Dimension Support:** Handles $d = 1, 2, 3$ dimensional lattices.
- **Multiorbital Lattices:** Fully supports arbitrary orbitals/sites per unit cell ($Q \ge 1$).
- **Flat Band Detector:** Grid-diagonalization of the Brillouin Zone (BZ) with variance detection.
- **Analytical CLS Extraction:** Uses Faddeev-LeVerrier trace recursion on Laurent polynomial matrices to calculate the exact, division-free adjugate matrix, avoiding $0/0$ singularities.
- **Symbolic GCD Reduction:** Employs SymPy to divide out common Laurent polynomial factors, minimizing real-space CLS support.
- **Topological Singularity Classifier:** Scans BZ candidate points and identifies the presence of immovable wave-function discontinuities (band crossings).
- **Degenerate Band Mixing ($M > 1$):** Handles degenerate flat bands (like Kagome-3) by attempting linear combination mixing ($w_0 \pm w_1$) to remove spurious individual singularities.
- **Numerical IDFT Cross-Validation:** Projects numerical BZ eigenstates onto the analytical CLS reference using norm/phase alignment, verifying numerical and analytical matching to machine precision ($10^{-16}$).
- **Non-Contractible States (NLS/NPS):** Automatically constructs 1D non-contractible loop states (NLS) or 2D non-contractible planar states (NPS) for singular flat bands.
- **Visualization:** Matplotlib plotting of band structures along high-symmetry BZ paths and real-space CLS/NLS lattice overlays.

---

## 2. Theoretical Background

### A. Finite Sum of Bloch Phases (FSBP)
A state $|\chi_R\rangle$ constructed as a linear combination of flat-band Bloch states is compact (localized to a finite region) if and only if its momentum-space wavefunction components $\alpha_k v_{k,q}$ satisfy the **FSBP condition**:
$$\alpha_k v_{k,q} = \sum_{m_1 \dots m_d} f^{(q)}_{m_1 \dots m_d} \exp\left( i \sum_l m_l k_l \cdot a_l \right)$$
This corresponds to a multi-variable **Laurent polynomial** in the variables $X_l = e^{i k_l \cdot a_l}$. The coefficients $f^{(q)}_{m}$ correspond directly to the real-space amplitudes of the CLS.

### B. Minor + Adjugate Method
For a modified Hamiltonian $\bar{H}_k = H_k - \epsilon_0 I$ at the flat band energy $\epsilon_0$, the unminimized CLS eigenvector $x_k$ is obtained by choosing a guide orbital $p$ and taking:
$$x_{k,p} = \text{det}(\bar{H}_k^{(p,p)})$$
$$x_{k, j \neq p} = \left[ - \text{Adj}(\bar{H}_k^{(p,p)}) \cdot h_p \right]_j$$
where $\bar{H}_k^{(p,p)}$ is the submatrix with row/column $p$ removed, and $h_p$ is the $p$-th column of $\bar{H}_k$.

### C. Singularity Classification
- **Non-singular:** The gauge factor $\alpha_k$ is non-zero everywhere in the BZ. The Bloch wavefunctions are smooth and continuous. The CLS translations form a complete basis.
- **Singular:** The gauge factor $\alpha_k$ vanishes at one or more points $k_0$ (immovable discontinuities/band-crossing points). Complete basis representation collapses; the missing degrees of freedom are recovered by NLS (2D) or NPS (3D).

---

## 3. Codebase Structure

The project has a modular Python package structure:

*   [__init__.py](file:///e:/Study/Topology/CLS-Calculator/cls_finder/__init__.py) & [__main__.py](file:///e:/Study/Topology/CLS-Calculator/cls_finder/__main__.py): Package entries.
*   [cli.py](file:///e:/Study/Topology/CLS-Calculator/cls_finder/cli.py): CLI interface parsing input JSONs and coordinating execution.
*   [report.py](file:///e:/Study/Topology/CLS-Calculator/cls_finder/report.py): JSON and human-readable text report generators.
*   [core/laurent.py](file:///e:/Study/Topology/CLS-Calculator/cls_finder/core/laurent.py): Multi-variable Laurent polynomial algebra (`LaurentPoly`).
*   [core/matrixpoly.py](file:///e:/Study/Topology/CLS-Calculator/cls_finder/core/matrixpoly.py): Matrices of Laurent polynomials (`MatrixPoly`) with trace, determinant, and Faddeev-LeVerrier adjugate solvers.
*   [core/lattice.py](file:///e:/Study/Topology/CLS-Calculator/cls_finder/core/lattice.py): Lattice coordinate translation and geometry mapping.
*   [io/parser.py](file:///e:/Study/Topology/CLS-Calculator/cls_finder/io/parser.py): Translates symbolic Hamiltonians or real-space hoppings to `MatrixPoly`.
*   [band/bands.py](file:///e:/Study/Topology/CLS-Calculator/cls_finder/band/bands.py): Uniform BZ grid diagonalizer and variance-based flat band detection.
*   [eigen/eigenstate.py](file:///e:/Study/Topology/CLS-Calculator/cls_finder/eigen/eigenstate.py): Analytical SymPy nullspace solver and numerical gauge alignment.
*   [classify/singularity.py](file:///e:/Study/Topology/CLS-Calculator/cls_finder/classify/singularity.py): BZ candidate point scanner, SVD rank scans, and degenerate band mixing.
*   [cls/analytic.py](file:///e:/Study/Topology/CLS-Calculator/cls_finder/cls/analytic.py): Minor + Adjugate analytical CLS calculation with nullspace fallback.
*   [cls/numeric.py](file:///e:/Study/Topology/CLS-Calculator/cls_finder/cls/numeric.py): Numerical CLS extraction using BZ eigenstate projection and IDFT.
*   [cls/reduce.py](file:///e:/Study/Topology/CLS-Calculator/cls_finder/cls/reduce.py): SymPy multivariable GCD minimization of Laurent polynomials.
*   [cls/noncontractible.py](file:///e:/Study/Topology/CLS-Calculator/cls_finder/cls/noncontractible.py): Constructs NLS/NPS along BZ slicing axes.
*   [viz/plot.py](file:///e:/Study/Topology/CLS-Calculator/cls_finder/viz/plot.py): Matplotlib plotting of band structures and real-space lattices.
*   [models/library.py](file:///e:/Study/Topology/CLS-Calculator/cls_finder/models/library.py): Built-in database of all 11 validation models.

---

## 4. Input Specification (JSON Schema)

The input configuration defines the lattice and the Hamiltonian. Two modes of Hamiltonian input are supported:

### Mode A: Real-Space Hoppings (Recommended)
```json
{
  "lattice": {
    "dimension": 2,
    "primitive_vectors": [[1.0, 0.0], [0.0, 1.0]],
    "orbitals": [
      {"label": "A", "position": [0.0, 0.0]},
      {"label": "B", "position": [0.5, 0.5]}
    ]
  },
  "hoppings": [
    {"i": 0, "j": 1, "R": [0, 0],  "t": -1.0},
    {"i": 0, "j": 1, "R": [-1, 0], "t": -1.0},
    {"i": 0, "j": 0, "R": [0, 0],  "t":  2.0}
  ],
  "options": {
    "k_grid": [40, 40],
    "flat_tol": 1e-5
  }
}
```
- `i`, `j`: Orbital indices ($0$-indexed).
- `R`: Integer displacement tuple (unit cell coordinates).
- `t`: Hopping amplitude (can be complex, e.g. `{"re": 0, "im": 1}`).

### Mode B: Symbolic k-space Hamiltonian
```json
{
  "lattice": {
    "dimension": 2,
    "primitive_vectors": [[1.0, 0.0], [0.0, 1.0]],
    "orbitals": [
      {"label": "A", "position": [0.0, 0.0]},
      {"label": "B", "position": [0.5, 0.5]}
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
```
- Variables must be written as `kx`, `ky`, `kz` and standard functions like `exp`, `cos`, `sin`, `I` are supported.

---

## 5. Usage

### Command Line Interface
Run the CLS Finder on any input JSON configuration:
```bash
python -m cls_finder run input.json --output_dir output
```

This will run the pipeline and generate:
- `output/band_structure.png`: Energy band structures.
- `output/cls_band_X.png`: Minimized CLS amplitude layout overlaying the lattice.
- `output/nls_band_X_axis_Y.png`: NLS/NPS layout (if singular).
- `output/report.json`: Machine-readable results.
- `output/report.txt`: Human-readable summary report.

---

## 6. Model Library

The model database [library.py](file:///e:/Study/Topology/CLS-Calculator/cls_finder/models/library.py) contains the 11 tight-binding models used to validate the program:

1.  **Zigzag Chain:** 1D chain ($d=1, Q=2$), non-singular, flat band at $E=-2.0$.
2.  **Kagome (NN hopping):** 2D lattice ($d=2, Q=3$), singular at $(0,0)$, flat band at $E=-2.0$.
3.  **Bilayer Square:** 2D lattice ($d=2, Q=2$), non-singular touching, flat band at $E=2.0$.
4.  **Lieb Lattice:** 2D lattice ($d=2, Q=3$), singular at $(\pi,\pi)$, flat band at $E=0.0$.
5.  **Modified Lieb Lattice:** 2D lattice ($d=2, Q=3$), singular at $(0,0)$, flat band at $E=0.0$.
6.  **Checkerboard Model-I:** 2D lattice ($d=2, Q=2$), singular at $(0,0)$, flat band at $E=0.0$.
7.  **Checkerboard Model-II:** 2D lattice ($d=2, Q=2$), singular at $(\pi,\pi)$, flat band at $E=0.0$.
8.  **Checkerboard Model-III:** 2D lattice ($d=2, Q=2$), non-singular, flat band at $E=0.0$.
9.  **Honeycomb Flat Model:** 2D lattice ($d=2, Q=2$), non-singular, flat band at $E=0.0$.
10. **Cubic 3D:** 3D lattice ($d=3, Q=3$), singular at 8 points, flat band at $E=0.0$.
11. **Kagome-3:** 2D lattice ($d=2, Q=3$), doubly degenerate ($M=2$), non-singular, flat band at $E=-2.0$.

---

## 7. Tests & Verification

The testing framework verifies both the core Laurent/matrix algebra and the physics pipeline.

To run the complete test suite:
```bash
python -m pytest -v
```

Tests include:
- `tests/test_laurent.py`: Basic polynomial operations, SymPy conversion, GCD algebra, and Faddeev-LeVerrier adjugate.
- `tests/test_pipeline.py`: End-to-end integration regression checks on all 11 library models verifying energy, singularity, null-vector checks, and IDFT cross-validation.
