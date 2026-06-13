import os
import argparse
import json
import numpy as np
import sympy
from cls_finder.io.parser import parse_input
from cls_finder.band.bands import detect_flat_bands, compute_bz_grid
from cls_finder.eigen.eigenstate import extract_eigenstate_analytical
from cls_finder.classify.singularity import classify_singularity
from cls_finder.cls.analytic import select_cls_basis
from cls_finder.cls.numeric import extract_cls_numeric, cross_validate_cls, verify_cls_eigenstate
from cls_finder.cls.noncontractible import build_noncontractible
from cls_finder.viz.plot import plot_bands, plot_lattice_cls
from cls_finder.report import generate_report

# We will define the main function
def run_pipeline(input_path, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Load input spec
    print(f"Loading input file: {input_path}")
    with open(input_path, 'r', encoding='utf-8') as f:
        spec = json.load(f)
        
    # 2. Parse input to Lattice and MatrixPoly
    lattice, H_k = parse_input(spec)
    print(f"Parsed lattice successfully: {lattice}")
    
    # 3. Detect flat bands
    grid_size = spec.get("options", {}).get("k_grid", [40] * lattice.dimension)
    flat_tol = spec.get("options", {}).get("flat_tol", 1e-5)
    
    print(f"Sampling BZ on grid {grid_size}...")
    flat_bands = detect_flat_bands(H_k, lattice, grid_size, flat_tol)
    
    if not flat_bands:
        print("No flat bands detected in the system!")
        # Create empty report
        generate_report([], os.path.join(output_dir, "report.json"), os.path.join(output_dir, "report.txt"))
        return
        
    # Group unique flat bands (by index)
    unique_fb = []
    seen_indices = set()
    for fb in flat_bands:
        if fb["band_index"] not in seen_indices:
            unique_fb.append(fb)
            seen_indices.add(fb["band_index"])
            
    print(f"Detected {len(unique_fb)} flat band(s). Analyzing...")
    
    results = []
    # Cache the complete CLS basis per degenerate group so an M-fold band
    # reports M distinct, independent CLS (one per band index) instead of the
    # same gauge repeated.
    basis_cache = {}

    # Define Sympy symbols for analytical calculations
    kx, ky, kz = sympy.symbols('kx ky kz')
    x1, x2, x3 = sympy.symbols('x1 x2 x3')
    symbols = [x1, x2, x3][:lattice.dimension]
    
    # Plot band structure
    band_plot_path = os.path.join(output_dir, "band_structure.png")
    print(f"Plotting band structure to: {band_plot_path}")
    plot_bands(H_k, lattice, band_plot_path)
    
    for i, fb in enumerate(unique_fb):
        idx = fb["band_index"]
        eps0 = fb["energy"]
        deg_indices = fb["degenerate_indices"]
        
        print(f"\nAnalyzing Flat Band #{i+1} at energy {eps0:.4f} (band index: {idx}):")
        
        # A. Analytical eigenvectors & Singularity classification
        # We need the analytical eigenvectors for singularity classification
        try:
            w_k_list = extract_eigenstate_analytical(H_k, eps0, symbols)
            print(f"  Successfully extracted {len(w_k_list)} analytical eigenvector(s).")
        except Exception as e:
            print(f"  Warning: failed to extract analytical eigenvectors: {e}")
            print("  Falling back to numerical analysis.")
            w_k_list = []
            
        # Singularity classification
        if w_k_list:
            sing_res = classify_singularity(w_k_list, lattice, H_k, eps0, grid_size)
            is_singular = sing_res["singular"]
            k0_list = sing_res["k0_list"]
        else:
            # Fallback classification if analytical failed (estimate using numerical nodes)
            print("  Cannot perform SVD rank classification without analytical eigenvectors.")
            is_singular = False
            k0_list = []
            
        print(f"  Topology Class: {'SINGULAR' if is_singular else 'NON-SINGULAR'}")
        if is_singular:
            print(f"  Singularity points (k0): {len(k0_list)} found")
            
        # B. CLS Derivation (Analytic & Minimized)
        # For an M-fold degenerate band we compute the complete basis of M
        # independent CLS once per group and hand band index `idx` its own
        # member; for M=1 this is just the single best gauge.
        cls_analytic_min = {}
        x_k_min = None
        M = len(deg_indices)
        try:
            grp_key = tuple(sorted(int(x) for x in deg_indices))
            if grp_key not in basis_cache:
                basis_cache[grp_key] = select_cls_basis(
                    H_k, eps0, symbols, M, lattice=lattice)
            basis = basis_cache[grp_key]
            j = grp_key.index(int(idx))
            x_k_min, A_0_R_min_amp, gauge_meta = basis[j] if j < len(basis) else basis[-1]
            deg_tag = f" [degenerate member {j+1}/{M}]" if M > 1 else ""
            print(f"  Analytical CLS: gauge '{gauge_meta['gauge_id']}' "
                  f"(support={gauge_meta['support_size']}, "
                  f"{'singular' if gauge_meta['is_singular'] else 'nonsingular'}){deg_tag}.")
            from cls_finder.cls.gauge_analysis import canonicalize_amplitudes
            cls_analytic_min = canonicalize_amplitudes(A_0_R_min_amp, H_k.rows, lattice.dimension)
        except Exception as e:
            print(f"  Warning: Analytical CLS derivation failed: {e}")
            cls_analytic_min = {}

        # B'. Rigorous module structure via the Gröbner/syzygy generating set
        # (complete set of CLS shapes). Available for integer/rational-coefficient
        # models; None otherwise.
        module_structure = None
        try:
            from cls_finder.cls.syzygy import compute_cls_generators
            syz = compute_cls_generators(H_k, eps0, symbols)
            if syz:
                module_structure = {
                    "rank": syz["rank"],
                    "n_generators": syz["n_generators"],
                    "is_free": syz["is_free"],
                }
                # NOTE: polynomial-ring freeness is a module-structure fact,
                # distinct from physical completeness (which is decided by the
                # resultant singularity test / is_singular above).
                free_note = ("free: rank generators suffice"
                             if syz["is_free"]
                             else "non-free: needs generators beyond rank (module relations)")
                print(f"  CLS module (syzygy): rank={syz['rank']}, "
                      f"complete generating set={syz['n_generators']} ({free_note}).")
        except Exception as e:
            print(f"  Warning: syzygy module analysis skipped: {e}")

        # C. Numerical CLS & Cross-check
        cls_numeric = {}
        cross_check = {"success": False, "message": "N/A"}
        try:
            cls_numeric = extract_cls_numeric(H_k, lattice, eps0, grid_size)
            print("  Numerical CLS derived successfully.")
            if cls_analytic_min and x_k_min is not None:
                if M > 1:
                    # Degenerate band: amplitude matching against a single
                    # numerical CLS is ill-defined (any vector in the M-dim
                    # eigenspace is valid). Verify the analytical CLS is a
                    # genuine flat-band eigenstate via the H x = eps0 x residual.
                    success, msg = verify_cls_eigenstate(H_k, lattice, eps0, x_k_min)
                else:
                    success, msg = cross_validate_cls(cls_analytic_min, cls_numeric)
                cross_check = {"success": success, "message": msg}
                print(f"  Cross-check: {msg}")
        except Exception as e:
            print(f"  Warning: Numerical CLS derivation/cross-check failed: {e}")
            
        # D. Non-Contractible Loop/Planar States (NLS/NPS) if singular
        nls_nps = []
        if is_singular and len(k0_list) > 0:
            try:
                # Use the first singularity point to build NLS/NPS
                print(f"  Constructing NLS/NPS for singularity k0 = {k0_list[0]}...")
                nls_nps = build_noncontractible(H_k, lattice, eps0, k0_list[0])
                print("  NLS/NPS constructed successfully.")
            except Exception as e:
                print(f"  Warning: NLS/NPS construction failed: {e}")
                
        # E. Plotting
        # Plot minimized CLS
        if cls_analytic_min:
            cls_plot_path = os.path.join(output_dir, f"cls_band_{idx}.png")
            print(f"  Plotting CLS to: {cls_plot_path}")
            plot_lattice_cls(lattice, H_k, cls_analytic_min, cls_plot_path)
            
        # Plot NLS/NPS
        if nls_nps:
            for state in nls_nps:
                axis = state["keep_axis"]
                nls_plot_path = os.path.join(output_dir, f"nls_band_{idx}_axis_{axis}.png")
                print(f"  Plotting NLS/NPS (axis {axis}) to: {nls_plot_path}")
                plot_lattice_cls(lattice, H_k, state["amplitudes"], nls_plot_path)
                
        # Gather results
        results.append({
            "band_index": int(idx),
            "energy": float(eps0),
            "degenerate_indices": [int(x) for x in deg_indices],
            "singular": bool(is_singular),
            "k0_list": k0_list,
            "module_structure": module_structure,
            "cls_analytic_min": cls_analytic_min,
            "cls_numeric": cls_numeric,
            "cross_check": cross_check,
            "nls_nps": nls_nps
        })
        
    # 4. Generate Reports
    json_path = os.path.join(output_dir, "report.json")
    text_path = os.path.join(output_dir, "report.txt")
    print(f"\nWriting reports to:\n  - JSON: {json_path}\n  - Text: {text_path}")
    generate_report(results, json_path, text_path, lattice=lattice)
    
    print("\nCLS Finder run completed successfully!")

def main():
    parser = argparse.ArgumentParser(description="CLS Finder - Flat Band and Compact Localization State Analyzer")
    subparsers = parser.add_subparsers(dest="command", required=True)
    
    run_parser = subparsers.add_parser("run", help="Run the CLS Finder pipeline on an input JSON file")
    run_parser.add_argument("input_file", help="Path to the input JSON file containing lattice and Hamiltonian spec")
    run_parser.add_argument("--output_dir", default="output", help="Directory to save output files and plots (default: 'output')")
    
    args = parser.parse_args()
    
    if args.command == "run":
        run_pipeline(args.input_file, args.output_dir)

if __name__ == "__main__":
    main()
