import json
import numpy as np

def convert_to_serializable(obj):
    """Recursively convert numpy arrays, tuples, complex numbers, etc. to JSON serializable objects."""
    if isinstance(obj, dict):
        # JSON keys must be strings
        new_dict = {}
        for k, v in obj.items():
            new_key = str(k)
            new_dict[new_key] = convert_to_serializable(v)
        return new_dict
    elif isinstance(obj, (list, tuple)):
        return [convert_to_serializable(x) for x in obj]
    elif isinstance(obj, np.ndarray):
        return convert_to_serializable(obj.tolist())
    elif isinstance(obj, complex):
        # Represent complex numbers as a dict or string
        if abs(obj.imag) < 1e-12:
            return float(obj.real)
        return {"re": float(obj.real), "im": float(obj.imag)}
    elif isinstance(obj, (np.int64, np.int32)):
        return int(obj)
    elif isinstance(obj, (np.float64, np.float32)):
        return float(obj)
    else:
        return obj

def generate_report(results, filepath_json, filepath_text, lattice=None):
    """
    Generate JSON and text reports.
    results: list of dict, one for each flat band analyzed:
    {
       "band_index": int,
       "energy": float,
       "degenerate_indices": list[int],
       "singular": bool,
       "k0_list": list[np.ndarray],
       "cls_analytic_min": dict,
       "cls_numeric": dict,
       "cross_check": dict,
       "nls_nps": list[dict]
    }
    """
    # 1. JSON Report
    serializable_results = convert_to_serializable(results)
    with open(filepath_json, 'w', encoding='utf-8') as f:
        json.dump(serializable_results, f, indent=2, ensure_ascii=False)
        
    # 2. Text Report
    lines = []
    lines.append("=" * 60)
    lines.append("               CLS FINDER - ANALYSIS SUMMARY")
    lines.append("=" * 60)
    lines.append("")
    
    if lattice is not None:
        from cls_finder.band.bands import get_reciprocal_vectors
        from cls_finder.viz.plot import get_symmetry_path
        B = get_reciprocal_vectors(lattice)
        points, labels = get_symmetry_path(lattice)
        
        lines.append("=" * 60)
        lines.append("            LATTICE RECIPROCAL SPACE INFO")
        lines.append("=" * 60)
        lines.append("Reciprocal Lattice Vectors B:")
        for idx, b_vec in enumerate(B):
            b_str = ", ".join(f"{x:.6f}" for x in b_vec)
            b_str_2pi = ", ".join(f"{x/(2.0*np.pi):.6f}" for x in b_vec)
            lines.append(f"  b_{idx+1} = [{b_str}]  (units of 2pi: [{b_str_2pi}])")
        lines.append("")
        lines.append("High Symmetry Points:")
        for p, label in zip(points, labels):
            plain_label = label.replace(r'$\Gamma$', 'Gamma').replace(r'$\pi$', 'pi').replace('$', '').replace('\\', '')
            cart_coord = p @ B
            cart_str = ", ".join(f"{x:.6f}" for x in cart_coord)
            cart_str_2pi = ", ".join(f"{x/(2.0*np.pi):.6f}" for x in cart_coord)
            frac_str = ", ".join(f"{x:.6f}" for x in p)
            lines.append(f"  {plain_label:<5} : Fractional = [{frac_str}] | Cartesian = [{cart_str}] (units of 2pi: [{cart_str_2pi}])")
        lines.append("")
        lines.append("=" * 60)
        lines.append("")

    lines.append(f"Total Flat Bands Analyzed: {len(results)}")
    lines.append("")
    
    for i, res in enumerate(results):
        lines.append("-" * 50)
        lines.append(f"Flat Band #{i+1} (Band Index: {res['band_index']})")
        lines.append("-" * 50)
        lines.append(f"  Energy (eps0): {res['energy']:.6f}")
        lines.append(f"  Degeneracy: {len(res['degenerate_indices'])} (Bands: {res['degenerate_indices']})")
        lines.append(f"  Topology Class: {'SINGULAR' if res['singular'] else 'NON-SINGULAR'}")
        
        if res['singular']:
            lines.append("  Singularity points (k0):")
            for idx, k0 in enumerate(res['k0_list']):
                k0_str = ", ".join(f"{x:.4f}" for x in k0)
                lines.append(f"    Point #{idx+1}: [{k0_str}]")
        else:
            lines.append("  Singularity points (k0): None (continuous Bloch wavefunctions)")
            
        lines.append("")
        
        # Cross-validation summary
        cc = res.get("cross_check", {})
        if cc:
            lines.append(f"  Numerical Cross-Check: {cc.get('message', 'N/A')}")
            lines.append(f"  Match Status: {'SUCCESS' if cc.get('success', False) else 'WARNING/FAIL'}")
        else:
            lines.append("  Numerical Cross-Check: N/A")
            
        lines.append("")
        
        # Minimized CLS support
        lines.append("  Minimized CLS Amplitude Support:")
        cls_min = res.get("cls_analytic_min", {})
        for q, cells in cls_min.items():
            if cells:
                lines.append(f"    Orbital {q}:")
                for cell, val in sorted(cells.items()):
                    cell_str = ",".join(str(x) for x in cell)
                    if isinstance(val, complex):
                        val_str = f"{val.real:.4f} + {val.imag:.4f}i" if abs(val.imag) > 1e-4 else f"{val.real:.4f}"
                    else:
                        val_str = f"{val:.4f}"
                    lines.append(f"      Cell ({cell_str}): amplitude = {val_str}")
                    
        if res['singular'] and res.get("nls_nps"):
            lines.append("")
            lines.append("  Non-Contractible Loop/Planar States (NLS/NPS) constructed:")
            for state in res["nls_nps"]:
                lines.append(f"    State along axis {state['keep_axis']} (compact along this axis, extended in others):")
                # Print sample amplitudes (first 5 elements to avoid spamming)
                cnt = 0
                for q, cells in state["amplitudes"].items():
                    for cell, val in sorted(cells.items()):
                        cell_str = ",".join(str(x) for x in cell)
                        if isinstance(val, complex):
                            val_str = f"{val.real:.3f} + {val.imag:.3f}i" if abs(val.imag) > 1e-4 else f"{val.real:.3f}"
                        else:
                            val_str = f"{val:.3f}"
                        # Print only if the coordinate in extended directions is 0 to show the 1D core
                        # i.e., all other coords are 0
                        is_core = True
                        for axis_idx, coord_val in enumerate(cell):
                            if axis_idx != state['keep_axis'] and coord_val != 0:
                                is_core = False
                                break
                        if is_core:
                            lines.append(f"      Orbital {q}, Cell ({cell_str}): {val_str}")
                            cnt += 1
                        if cnt >= 5:
                            break
                    if cnt >= 5:
                        lines.append("      ... (and plane-wave extended along other unit cells)")
                        break
        lines.append("")
        
    lines.append("=" * 60)
    lines.append("                  END OF ANALYSIS REPORT")
    lines.append("=" * 60)
    
    text_content = "\n".join(lines)
    with open(filepath_text, 'w', encoding='utf-8') as f:
        f.write(text_content)
        
    return text_content
