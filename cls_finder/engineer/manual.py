"""
Module 6 -- Manual Real-Space CLS Placement.

Lets the user build f(k) directly by toggling individual real-space CLS sites
(sublattice alpha, cell offset (n, m), amplitude A, phase theta) on the
real-space lattice picture, instead of going through the (C, w_i)-target
pairing/chiral pipeline (Modules 2/3, pairing.py / chiral.py).

H(k) = E0 I + alpha*H_core(k) + H_core(k) M_tilde(k) H_core(k)
(cls_finder.engineer.hamiltonian.build_engineered_hamiltonian, "Hamiltonian
Engineering" note Sec.2) guarantees H(k) is EXACTLY flat at E0 for ANY f(k)
and ANY (alpha, M_tilde) -- so there is no separate "flatness check" here, and
no analytic patching at the zeros of f(k) is needed (H_core(k) is itself a
finite polynomial, well-defined everywhere, including at f(k)=0 where
H_core(k_i)=0). ``analyze_manual_cls`` auto-discovers the common zeros of the
user's f(k) (``chern.find_common_zeros``, no pre-specified target) and runs
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
from cls_finder.engineer.hamiltonian import (
    build_engineered_hamiltonian, matrixpoly_to_hoppings, max_hopping_range,
    flatness_deviation, hoppings_to_matrixpoly,
)
from cls_finder.engineer.dispersive import build_dispersive_M


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
    """Sanity-check a manually-built f(k). No flatness check is needed --
    build_engineered_hamiltonian guarantees H(k) is flat at E0 for ANY f(k).

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
                       E0: float = 0.0, alpha: float = 1.0,
                       dispersive_shape: str = "nn_real",
                       dispersive_strength: float = 0.3,
                       M_tilde: Optional[MatrixPoly] = None,
                       r_max: Optional[int] = None,
                       scan_n: int = 160, grid_n: int = 24,
                       loop_n: int = 128) -> Dict:
    """Build f(k) from user-placed sites, construct the always-exactly-flat
    H(k) = E0 I + alpha*H_core(k) + H_core(k) M_tilde(k) H_core(k), auto-discover
    its topology (no pre-specified target), and produce (optionally truncated)
    real-space hoppings.

    Returns a dict:
      valid, reason   : False/short-circuit if f(k) == 0 (see validate_manual_cls)
      trivial, warnings
      x_k             : list[LaurentPoly]
      zeros           : list[np.ndarray] -- common zeros of f(k) (chern.find_common_zeros)
      H_full          : MatrixPoly -- exact, untruncated H(k) (always flat at E0)
      chern_report    : chern.explore_brillouin_zone(...) -- full Chern/topology report
      ift, H_trunc, iso_trunc, num_trunc : (optionally r_max-truncated) real-space model
      natural_hopping_range, max_hopping_range, flat_band_max_dev, exact_flat,
      r_max, r_max_applied
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

    M_tilde_resolved = M_tilde if M_tilde is not None else build_dispersive_M(
        lattice_spec, dispersive_shape, dispersive_strength)
    H_full = build_engineered_hamiltonian(x_k, E0=E0, alpha=alpha, M_tilde=M_tilde_resolved)

    chern_report = _chern.explore_brillouin_zone(
        H_full, lattice, E0, 1, x_k=x_k, grid_n=grid_n, scan_n=scan_n, loop_n=loop_n)
    log.append(f"[Manual] 위상 분석: C={chern_report['chern_number']} "
               f"(well_defined={chern_report['well_defined']})")

    hoppings_full = matrixpoly_to_hoppings(H_full)
    natural_range = max_hopping_range(hoppings_full)
    weight_full = sum(abs(v) ** 2 for v in hoppings_full.values())

    r_max_applied = False
    if r_max is not None and natural_range > r_max:
        hoppings = {key: v for key, v in hoppings_full.items()
                    if max(abs(key[2][0]), abs(key[2][1])) <= r_max}
        H_trunc = hoppings_to_matrixpoly(hoppings, lattice_spec.N)
        r_max_applied = True
        weight_kept = sum(abs(v) ** 2 for v in hoppings.values())
        trunc_ratio = 1.0 - (weight_kept / weight_full if weight_full > 1e-300 else 1.0)
    else:
        hoppings, H_trunc, trunc_ratio = hoppings_full, H_full, 0.0

    max_range = max_hopping_range(hoppings)
    flat_dev = flatness_deviation(H_trunc, x_k, lattice_spec, E0)
    iso_trunc = _chern.band_isolation(H_trunc, lattice, E0, 1)
    num_trunc = _chern.numerical_chern(H_trunc, lattice, E0, 1)
    ift = {"hoppings": hoppings, "R_cut": max_range, "truncation_ratio": trunc_ratio}

    if r_max_applied:
        log.append(f"[Manual] r_max={r_max}로 절단 (natural range {natural_range}): "
                   f"{len(hoppings)}개 항, {trunc_ratio*100:.2f}% ||t||^2 제거 -> "
                   f"flatness max||H psi-E0 psi||={flat_dev:.2e} (근사: 절단으로 "
                   "완전한 평탄성이 깨질 수 있음)")
    else:
        log.append(f"[Manual] 정확한 모델: {len(hoppings)}개 hopping 항, 최대 거리 "
                   f"{max_range} 셀, flatness max||H psi-E0 psi||={flat_dev:.2e} "
                   "(정확함, 절단 없음)")
    log.append(f"[Manual] H_trunc isolated={iso_trunc['isolated']}, "
               f"numerical C_trunc={num_trunc['C']}")

    return {
        "valid": True, "trivial": val["trivial"], "warnings": val["warnings"],
        "x_k": x_k, "zeros": zeros, "H_full": H_full, "chern_report": chern_report,
        "ift": ift, "H_trunc": H_trunc, "iso_trunc": iso_trunc, "num_trunc": num_trunc,
        "natural_hopping_range": natural_range, "max_hopping_range": max_range,
        "flat_band_max_dev": flat_dev, "exact_flat": flat_dev < 1e-6,
        "r_max": r_max, "r_max_applied": r_max_applied,
        "log": log,
    }
