"""
Robust Boundary Mode — matplotlib visualization.

Two figures per the RBM spec:
  1. 2D (or 3D) real-space amplitude map: faint full lattice grid + markers on the
     boundary sites, marker size ∝ |psi|, colour = phase (cyclic) or sign.
  2. Skin-depth profile: max |psi| per layer along an axis (step function).
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from cls_finder.rbm.boundary_mode import boundary_mode_sites, skin_depth_profile


def _all_grid_positions(lattice, N):
    """Cartesian positions of every site in the N_1 x ... x N_d open lattice."""
    from itertools import product as iproduct
    d = lattice.dimension
    Q = lattice.num_orbitals
    pts = []
    for cell in iproduct(*[range(N[l]) for l in range(d)]):
        for q in range(Q):
            pts.append(lattice.get_cartesian_position(cell, q))
    return np.array(pts) if pts else np.zeros((0, lattice.spatial_dim))


def plot_amplitude_map(lattice, rbm_out, ax=None, amp_tol=1e-9,
                       size_scale=400.0):
    """2D real-space amplitude map. Returns the matplotlib Axes."""
    N = rbm_out["system_size"]
    if ax is None:
        _, ax = plt.subplots(figsize=(6.5, 6.0))

    grid = _all_grid_positions(lattice, N)
    if grid.shape[0]:
        gx = grid[:, 0]
        gy = grid[:, 1] if grid.shape[1] >= 2 else np.zeros_like(gx)
        ax.scatter(gx, gy, s=6, c="#d9dde3", marker=".", zorder=1,
                   linewidths=0)

    sites = [s for s in rbm_out["sites"] if s["abs"] >= amp_tol]
    if sites:
        xs = np.array([s["x"] for s in sites])
        ys = np.array([s["y"] for s in sites])
        mags = np.array([s["abs"] for s in sites])
        phases = np.array([s["phase"] for s in sites])
        mmax = mags.max() if mags.size else 1.0
        sizes = size_scale * (mags / mmax) + 12.0

        # Real CLS -> red/blue by sign; complex -> cyclic phase colormap.
        amps_im = np.array([abs(s["amp_im"]) for s in sites])
        is_real = float(amps_im.max()) < 1e-9 * max(mmax, 1e-12)
        if is_real:
            signs = np.array([np.sign(s["amp_re"]) for s in sites])
            colors = np.where(signs >= 0, "#d62728", "#1f77b4")
            sc = ax.scatter(xs, ys, s=sizes, c=colors, alpha=0.85,
                            edgecolors="k", linewidths=0.4, zorder=3)
        else:
            sc = ax.scatter(xs, ys, s=sizes, c=phases, cmap="twilight",
                            vmin=-np.pi, vmax=np.pi, alpha=0.9,
                            edgecolors="k", linewidths=0.4, zorder=3)
            cb = ax.figure.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)
            cb.set_label("phase  arg(ψ)")

    v = rbm_out["validation"]
    status = "BULK CANCELLED ✓" if v["passed"] else "bulk NOT cancelled ✗"
    sing = rbm_out.get("singular")
    sing_txt = ("singular" if sing else "nonsingular") if sing is not None else "?"
    ax.set_title(f"Robust Boundary Mode  ({sing_txt})\n"
                 f"{status}   max|ψ|_bulk={v['max_bulk_amp']:.1e}   "
                 f"radius={rbm_out['support_radius']}")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_aspect("equal", adjustable="datalim")
    return ax


def plot_skin_depth(rbm_out, axis=0, ax=None, transverse_bulk=True):
    """Step-function penetration profile along `axis`. Returns the Axes."""
    if ax is None:
        _, ax = plt.subplots(figsize=(5.5, 4.0))
    prof = (rbm_out["skin_depth"].get(axis)
            if "skin_depth" in rbm_out else None)
    if prof is None or prof.get("transverse_bulk") != transverse_bulk:
        prof = skin_depth_profile(rbm_out["_result"], axis=axis,
                                  transverse_bulk=transverse_bulk)
    dist = prof["distance"]
    vals = prof["max_amp"]
    ax.step(dist, vals, where="mid", color="#7c3aed", lw=2)
    ax.fill_between(dist, vals, step="mid", alpha=0.25, color="#7c3aed")
    ax.set_xlabel(f"layer index along axis {axis}")
    ax.set_ylabel(r"$\max_{\perp,q}\,|\psi_\partial|$")
    ax.set_title("Skin depth (step, not exponential)")
    ax.set_ylim(bottom=0)
    ax.grid(alpha=0.3)
    return ax


def plot_rbm(lattice, rbm_out, save_path=None, axis=0):
    """Combined figure: amplitude map + skin-depth profile. Saves if save_path."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.5, 5.6),
                                    gridspec_kw={"width_ratios": [1.4, 1.0]})
    plot_amplitude_map(lattice, rbm_out, ax=ax1)
    plot_skin_depth(rbm_out, axis=axis, ax=ax2)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=130, bbox_inches="tight")
        plt.close(fig)
        return save_path
    return fig
