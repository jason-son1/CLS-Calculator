"""
Module 6 -- Manual Real-Space CLS Placement.

Lets the user build f(k) directly by toggling individual real-space CLS sites
(sublattice alpha, cell offset (n, m), amplitude A, phase theta) on the
real-space lattice picture, instead of going through the (C, w_i)-target
pairing/chiral pipeline (Modules 2/3, pairing.py / chiral.py).

Eq. 127 (H(k) = E0 P(k) + (I-P(k)) M(k) (I-P(k))) guarantees H(k) is exactly
flat at E0 for ANY f(k) -- so there is no separate "flatness check" here.
Instead, ``analyze_manual_cls`` auto-discovers the common zeros of the user's
f(k) (``chern.find_common_zeros``, no pre-specified target) and runs
``chern.explore_brillouin_zone`` for an honest topology report (Chern number,
isolation, per-zero windings).
"""
from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np

from cls_finder.core.laurent import LaurentPoly
from cls_finder.core.matrixpoly import MatrixPoly
from cls_finder.classify import chern as _chern
from cls_finder.engineer.spec import LatticeSpec
from cls_finder.engineer.assembly import vortex_vector
from cls_finder.engineer.hamiltonian import (
    default_M_k, NumericHk, inverse_fourier_transform, hoppings_to_matrixpoly,
)


def build_x_k_from_sites(lattice_spec: LatticeSpec, cls_sites: List[Dict]) -> List[LaurentPoly]:
    """f(k) = [f_1(k), ..., f_N(k)] (Eq. 71) assembled directly from
    user-placed real-space CLS sites, bypassing pairing.py / chiral.py.

    cls_sites: [{"alpha": int, "n": int, "m": int, "A": float, "theta": float}, ...]
    Each site contributes c_{alpha,(n,m)} += A * e^{i(theta + zeta_alpha)} --
    the same coefficient convention as assembly.build_x_k (Eq. 71's
    per-sublattice gauge phase zeta^(alpha)).
    """
    N = lattice_spec.N
    coefs: List[Dict] = [dict() for _ in range(N)]
    for site in cls_sites:
        alpha = int(site["alpha"])
        if not (0 <= alpha < N):
            raise ValueError(f"CLS site sublattice index {alpha} out of range [0, {N})")
        n, m = int(site["n"]), int(site["m"])
        A = float(site["A"])
        theta = float(site["theta"])
        zeta = lattice_spec.sublattices[alpha].zeta
        c = A * np.exp(1j * (theta + zeta))
        key = (n, m)
        coefs[alpha][key] = coefs[alpha].get(key, 0j) + c
    return [LaurentPoly(c, d=2) for c in coefs]


def validate_manual_cls(x_k: List[LaurentPoly]) -> Dict:
    """Sanity-check a manually-built f(k). No flatness check is needed -- Eq.
    127 guarantees H(k) is flat at E0 for ANY f(k).

    valid=False (hard error) only if f(k) is identically zero (no sites
    placed / all amplitudes ~0 -- P(k) = f f^dagger / ||f||^2 is undefined
    everywhere).

    Otherwise valid=True, with a `trivial` flag and `warnings` list:
    trivial=True means every nonzero coefficient sits at the central cell
    (n, m) = (0, 0), so f(k) is k-independent -- P(k) is then a CONSTANT
    projector and the resulting flat band is a decoupled, topologically
    trivial (C=0) "molecular orbital" mode. Place sites in neighboring cells
    for nontrivial topology.
    """
    if all(p.is_zero() for p in x_k):
        return {"valid": False,
                "reason": "f(k)가 항등적으로 0입니다 (배치된 CLS 사이트가 없음). "
                          "격자 미리보기에서 사이트를 클릭해 CLS를 구성하세요."}

    nonconstant = any(any(exp != (0, 0) for exp in p.coefs) for p in x_k)
    trivial = not nonconstant
    warnings: List[str] = []
    if trivial:
        warnings.append(
            "모든 CLS 사이트가 중심 셀 (0,0)에만 있어 f(k)가 k에 무관합니다. "
            "P(k)가 상수 사영자가 되므로 결과 평탄 밴드는 위상이 자명한(C=0) "
            "분리된 분자궤도(molecular-orbital) 모드입니다. 비자명한 위상을 "
            "얻으려면 이웃 셀에도 사이트를 배치하세요."
        )
    return {"valid": True, "trivial": trivial, "warnings": warnings}


def analyze_manual_cls(lattice_spec: LatticeSpec, cls_sites: List[Dict],
                       E0: float = 0.0, t: float = 0.3, delta: float = 0.5,
                       M_k: Optional[MatrixPoly] = None,
                       n_grid_ift: int = 24, R_cut: int = 3,
                       max_rcut_retries: int = 4,
                       scan_n: int = 160, grid_n: int = 24,
                       loop_n: int = 128, zero_tol: float = 1e-8) -> Dict:
    """Build f(k) from user-placed sites, construct the (always-flat, Eq.127)
    H(k), auto-discover its topology (no pre-specified target), and produce
    truncated real-space hoppings.

    Returns a dict:
      valid, reason   : False/short-circuit if f(k) == 0 (see validate_manual_cls)
      trivial, warnings
      x_k             : list[LaurentPoly]
      zeros           : list[np.ndarray] -- common zeros of f(k) (chern.find_common_zeros)
      H_k             : hamiltonian.NumericHk (Eq. 127, exact -- always flat at E0)
      chern_report    : chern.explore_brillouin_zone(...) -- full Chern/topology report
      ift, H_trunc, iso_trunc, num_trunc, R_cut : truncated real-space model
      log             : list[str] human-readable progress lines
    """
    log: List[str] = []
    x_k = build_x_k_from_sites(lattice_spec, cls_sites)
    val = validate_manual_cls(x_k)
    if not val["valid"]:
        return {"valid": False, "reason": val["reason"], "x_k": x_k, "log": log}

    log.append(f"[Manual] {len(cls_sites)}개 CLS 사이트로 f(k) 구성 "
               f"(N={lattice_spec.N} 서브격자)")
    for w in val["warnings"]:
        log.append(f"[Manual][WARNING] {w}")

    lattice = lattice_spec.to_lattice()

    zeros = _chern.find_common_zeros(x_k, lattice, n_scan=scan_n)
    log.append(f"[Manual] f(k)의 공통 영점 {len(zeros)}개 발견")

    vortex_vectors = []
    for k_i in zeros:
        try:
            v = vortex_vector(x_k, k_i, lattice_spec.primitive_vectors)
            vortex_vectors.append((k_i, v))
        except ValueError:
            log.append(f"[Manual][NOTE] k={np.round(k_i, 4).tolist()}에서 1차 자코비안이 "
                       f"0 -- 고차 영점일 수 있어 해당 점의 P(k) 해석적 패치를 생략합니다")

    M = M_k if M_k is not None else default_M_k(lattice_spec, t=t, delta=delta)
    H_k = NumericHk(x_k, M, E0, lattice_spec, vortex_vectors, zero_tol=zero_tol)

    chern_report = _chern.explore_brillouin_zone(
        H_k, lattice, E0, 1, x_k=x_k, grid_n=grid_n, scan_n=scan_n, loop_n=loop_n)
    log.append(f"[Manual] 위상 분석: C={chern_report['chern_number']} "
               f"(well_defined={chern_report['well_defined']})")

    ift: Dict = {}
    H_trunc = None
    iso_trunc: Dict = {}
    num_trunc: Dict = {}
    rc = R_cut
    for rattempt in range(max_rcut_retries):
        rc = R_cut + rattempt
        ift = inverse_fourier_transform(H_k, lattice_spec, n_grid=n_grid_ift, R_cut=rc)
        H_trunc = hoppings_to_matrixpoly(ift["hoppings"], lattice_spec.N)
        iso_trunc = _chern.band_isolation(H_trunc, lattice, E0, 1)
        num_trunc = _chern.numerical_chern(H_trunc, lattice, E0, 1)
        log.append(f"[Manual] IFT R_cut={rc}: truncation_ratio="
                   f"{ift['truncation_ratio']:.2e}, H_trunc isolated="
                   f"{iso_trunc['isolated']}, numerical C_trunc={num_trunc['C']}")
        if iso_trunc["isolated"]:
            break
        log.append(f"[Manual][WARNING] R_cut={rc}: H_trunc가 고립되지 않음 -- R_cut 증가")

    return {
        "valid": True, "trivial": val["trivial"], "warnings": val["warnings"],
        "x_k": x_k, "zeros": zeros, "H_k": H_k, "chern_report": chern_report,
        "ift": ift, "H_trunc": H_trunc, "iso_trunc": iso_trunc, "num_trunc": num_trunc,
        "R_cut": rc, "log": log,
    }
