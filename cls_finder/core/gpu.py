"""
GPU acceleration helpers using CuPy.
Falls back to NumPy silently if CuPy is not installed or no GPU is available.
"""
import os
import sys

# Windows-specific DLL directory loading for pip-installed nvidia packages
if sys.platform == "win32":
    site_packages_dirs = []
    try:
        import site
        if hasattr(site, "getusersitepackages"):
            site_packages_dirs.append(site.getusersitepackages())
    except Exception:
        pass
    for path in sys.path:
        if "site-packages" in path:
            site_packages_dirs.append(path)
            
    # Try adding nvidia wheel dll directories
    _dll_handles = []
    for sp in set(site_packages_dirs):
        nvidia_path = os.path.join(sp, "nvidia")
        if os.path.exists(nvidia_path):
            for folder in os.listdir(nvidia_path):
                bin_path = os.path.join(nvidia_path, folder, "bin")
                if os.path.exists(bin_path):
                    try:
                        handle = os.add_dll_directory(bin_path)
                        _dll_handles.append(handle)
                    except Exception:
                        pass

try:
    import cupy as cp
    # Verify a GPU is actually usable and all libraries (cublas, cusolver) load fine
    cp.array([1.0])
    # Force loading of linear algebra modules to verify cusolver DLL loads successfully
    cp.linalg.eigvalsh(cp.eye(2))
    USE_GPU = True
    print("[CLS Finder] GPU (CuPy) acceleration enabled")
except Exception as e:
    USE_GPU = False
    print(f"[CLS Finder] GPU (CuPy) acceleration disabled or unavailable: {e}")


def eigvalsh_batch(H_batch):
    """Hermitian eigenvalues: (N, Q, Q) -> (N, Q). GPU if available."""
    if USE_GPU:
        import cupy as cp
        return cp.linalg.eigvalsh(cp.asarray(H_batch)).get()
    import numpy as np
    return np.linalg.eigvalsh(H_batch)


def eigh_batch(H_batch):
    """Hermitian eigen-decomposition: (N, Q, Q) -> ((N, Q), (N, Q, Q)). GPU if available."""
    if USE_GPU:
        import cupy as cp
        vals, vecs = cp.linalg.eigh(cp.asarray(H_batch))
        return vals.get(), vecs.get()
    import numpy as np
    return np.linalg.eigh(H_batch)


def idft_matmul(x_k_num, xi_points, m_arr_all):
    """
    Vectorized IDFT core:
        all_vals = (x_k_num.T @ exp(-2πi * xi @ m.T)) / N_total
    Returns a (Q, M_total) numpy array.
    GPU if available (biggest win for large 3D grids).
    """
    import numpy as np
    N_total = len(xi_points)
    if USE_GPU:
        import cupy as cp
        xi_gpu = cp.asarray(xi_points)
        m_gpu  = cp.asarray(m_arr_all)
        xk_gpu = cp.asarray(x_k_num)
        phases = cp.exp(-2j * cp.pi * (xi_gpu @ m_gpu.T))
        return (xk_gpu.T @ phases).get() / N_total
    else:
        phases = np.exp(-2j * np.pi * (xi_points @ m_arr_all.T))
        return (x_k_num.T @ phases) / N_total
